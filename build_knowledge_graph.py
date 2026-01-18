import json
import asyncio
import networkx as nx
import re
import uuid
import numpy as np
import hashlib
from doris_vector_search import DorisVectorClient, AuthOptions, IndexOptions

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from conf import settings
from rag_lib import get_embedding_model
from llm_adapters import get_async_chat_llm

class GraphExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.tuple_delimiter = "<|>"
        self.record_delimiter = "\n"

    async def extract(self, text: str) -> nx.Graph:
        # Construct Prompt to guide LLM to output triplets in a specific format
        prompt = self._build_prompt(text)
        response = await self.llm.chat(prompt)
        print(f"[Extractor] Raw response length: {len(response)}")
        return self._parse_response(response)

    def _build_prompt(self, text: str) -> str:
        return f"""
        -Goal-
        Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

        -Steps-
        1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_type: One of the following types: [Organization, Person, Location, Event, Concept]
        - entity_description: Comprehensive description of the entity's attributes and activities
        Format: ("entity"{self.tuple_delimiter}<entity_name>{self.tuple_delimiter}<entity_type>{self.tuple_delimiter}<entity_description>)

        2. From the entities identified in step 1, identify all relationships. For each relationship, extract the following information:
        - source_entity: Name of the source entity
        - target_entity: Name of the target entity
        - relationship_description: Explanation as to why you think the source entity and the target entity are related
        - relationship_strength: A numeric score indicating strength of the relationship between the source entity and target entity
        Format: ("relationship"{self.tuple_delimiter}<source_entity>{self.tuple_delimiter}<target_entity>{self.tuple_delimiter}<relationship_description>{self.tuple_delimiter}<relationship_strength>)

        -Output Requirement-
        Output ONLY the list of entities and relationships in the specified format. Do not include any other text, explanations, or markdown code blocks.

        -Input-
        {text}
        """

    def _parse_response(self, response: str) -> nx.Graph:
        G = nx.Graph()
        # Remove potential Markdown code block markers
        response = re.sub(r'^```.*?\n', '', response, flags=re.MULTILINE)
        response = re.sub(r'```$', '', response, flags=re.MULTILINE)
        
        lines = response.strip().split(self.record_delimiter)
        for line in lines:
            line = line.strip()
            if not line.startswith("(") or not line.endswith(")"):
                continue
            content = line[1:-1] # Remove leading and trailing parentheses
            parts = content.split(self.tuple_delimiter)
            
            # Parse Entity
            if len(parts) >= 4 and parts[0] in ['"entity"', "'entity'", "entity"]:
                name = parts[1]
                type_ = parts[2]
                desc = parts[3]
                G.add_node(name, type=type_, description=desc)
            
            # Parse Relationship
            elif len(parts) >= 5 and parts[0] in ['"relationship"', "'relationship'", "relationship"]:
                source = parts[1]
                target = parts[2]
                desc = parts[3]
                # Handle potential weight field
                weight = 1
                if len(parts) >= 5:
                     # Some models might put strength at the end, others might not
                     try:
                         weight = float(parts[-1])
                     except ValueError:
                         pass
                G.add_edge(source, target, description=desc, weight=weight)
        return G


class GraphStore:
    def __init__(self):
        doris_conf = settings.doris
        self.db_name = doris_conf.get('db_name', 'doris_rag_db')
        self.table_chunks = doris_conf.get('graph_chunks_table', doris_conf.get('table_name', 'graph_chunks'))
        self.table_doc_status = doris_conf.get('doc_status_table', 'doc_status')

        auth = AuthOptions(
            host=doris_conf.get('host', 'localhost'),
            query_port=int(doris_conf.get('query_port', 9030)),
            http_port=int(doris_conf.get('http_port', 8030)),
            user=doris_conf.get('user', 'root'),
            password=doris_conf.get('password', ''),
        )

        self.client = DorisVectorClient(self.db_name, auth_options=auth)
        print("[Store] doris-vector-search SDK initialized.")

        self.embed_model = get_embedding_model()
        self._init_tables()

    def _init_tables(self):
        try:
            conn = self.client.connection
            cursor = conn.cursor()
            
            # 1. Graph Chunks Table
            # Must be DUPLICATE KEY for ANN Index support in this Doris version
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_chunks}")
            sql_chunks = f"""
            CREATE TABLE IF NOT EXISTS {self.table_chunks} (
                id VARCHAR(128),
                kb_id VARCHAR(64),
                source_id VARCHAR(256),
                knowledge_graph_kwd VARCHAR(32),
                entity_kwd VARCHAR(256),
                from_entity_kwd VARCHAR(256),
                to_entity_kwd VARCHAR(256),
                content TEXT,
                embedding ARRAY<FLOAT> NOT NULL,
                INDEX idx_embedding (embedding) USING ANN PROPERTIES("dim"="1024", "index_type" = "hnsw", "metric_type"="inner_product")
            )
            ENGINE=OLAP
            DUPLICATE KEY(id)
            DISTRIBUTED BY HASH(id) BUCKETS 1
            PROPERTIES (
                "replication_num" = "1"
            );
            """
            cursor.execute(sql_chunks)

            # 2. Doc Status Table
            # Use UNIQUE KEY for Upsert semantics (no Vector Index needed here)
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_doc_status}")
            sql_status = f"""
            CREATE TABLE IF NOT EXISTS {self.table_doc_status} (
                doc_id VARCHAR(128),
                kb_id VARCHAR(64),
                doc_hash VARCHAR(64),
                status VARCHAR(16),
                updated_at DATETIME
            )
            ENGINE=OLAP
            UNIQUE KEY(doc_id, kb_id)
            DISTRIBUTED BY HASH(doc_id) BUCKETS 1
            PROPERTIES (
                "replication_num" = "1"
            );
            """
            cursor.execute(sql_status)
            
            while cursor.nextset():
                pass
            cursor.close()
            print("[Store] Tables ensured via SDK connection.")
        except Exception as e:
            print(f"[Store] Warning: Failed to init tables via SDK connection: {e}")

    def check_doc_status(self, doc_id: str, kb_id: str, content: str) -> str:
        """
        Check document status (Using SQL via SDK connection)
        Returns: 'new', 'updated', 'skipped'
        """
        current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        sql = f"SELECT doc_hash FROM {self.table_doc_status} WHERE doc_id=%s AND kb_id=%s"
        try:
            cursor = self.client.connection.cursor(dictionary=True)
            cursor.execute(sql, (doc_id, kb_id))
            res = cursor.fetchone()
            cursor.close()
            
            if not res:
                return 'new'
            
            if res['doc_hash'] != current_hash:
                return 'updated'
            
            return 'skipped'
        except Exception as e:
            print(f"[Store] Status check failed: {e}")
            return 'new'

    def update_doc_status(self, doc_id: str, kb_id: str, content: str):
        current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        # Use SQL for Upsert on Unique Key table
        sql = f"""
            INSERT INTO {self.table_doc_status} (doc_id, kb_id, doc_hash, status, updated_at)
            VALUES (%s, %s, %s, 'processed', NOW())
        """
        try:
            cursor = self.client.connection.cursor()
            cursor.execute(sql, (doc_id, kb_id, current_hash))
            cursor.close()
        except Exception as e:
            print(f"[Store] Error updating status: {e}")

    def delete_doc_data(self, doc_id: str, kb_id: str):
        """Delete old data corresponding to the document"""
        # SDK currently doesn't expose explicit delete easily in high-level API
        # For this demo, we skip deletion or we could implement it if SDK supports it.
        print(f"[Store] Skipping delete for doc {doc_id} (SDK limitation in this demo).")
        pass

    def save_subgraph(self, graph: nx.Graph, doc_id: str, kb_id: str):
        print(f"[Store] Saving subgraph for doc {doc_id}...")
        chunks = []
        
        # 1. Subgraph Snapshot
        chunks.append({
            "id": str(uuid.uuid4()),
            "kb_id": kb_id,
            "source_id": doc_id,
            "knowledge_graph_kwd": "subgraph",
            "entity_kwd": "",
            "from_entity_kwd": "",
            "to_entity_kwd": "",
            "content": json.dumps(nx.node_link_data(graph)),
            "text_to_embed": None
        })

        # 2. Entities
        for node, data in graph.nodes(data=True):
            desc = data.get('description', '')
            text_to_embed = f"{node}: {desc}"
            chunks.append({
                "id": str(uuid.uuid4()),
                "kb_id": kb_id,
                "source_id": doc_id,
                "knowledge_graph_kwd": "entity",
                "entity_kwd": node,
                "from_entity_kwd": "",
                "to_entity_kwd": "",
                "content": json.dumps(data),
                "text_to_embed": text_to_embed
            })

        # 3. Relations
        for u, v, data in graph.edges(data=True):
            desc = data.get('description', '')
            text_to_embed = f"{u} -> {v}: {desc}"
            chunks.append({
                "id": str(uuid.uuid4()),
                "kb_id": kb_id,
                "source_id": doc_id,
                "knowledge_graph_kwd": "relation",
                "entity_kwd": "",
                "from_entity_kwd": u,
                "to_entity_kwd": v,
                "content": json.dumps(data),
                "text_to_embed": text_to_embed
            })

        # Batch calculate Embeddings
        texts_to_embed = [c["text_to_embed"] for c in chunks if c["text_to_embed"]]
        if texts_to_embed:
            print(f"[Store] Generating embeddings for {len(texts_to_embed)} items...")
            embeddings = self.embed_model.embed_documents(texts_to_embed)
            
            embed_iter = iter(embeddings)
            for c in chunks:
                if c["text_to_embed"]:
                    c["embedding"] = next(embed_iter)
                else:
                    c["embedding"] = [0.0] * 1024

        # Insert via SDK
        data_to_insert = []
        for c in chunks:
            item = {
                "id": c["id"],
                "kb_id": c["kb_id"],
                "source_id": c["source_id"],
                "knowledge_graph_kwd": c["knowledge_graph_kwd"],
                "entity_kwd": c["entity_kwd"],
                "from_entity_kwd": c["from_entity_kwd"],
                "to_entity_kwd": c["to_entity_kwd"],
                "content": c["content"],
                "embedding": c["embedding"]
            }
            data_to_insert.append(item)
        
        try:
            table = self.client.open_table(self.table_chunks)
            table.add(data_to_insert)
            print(f"[Store] Inserted {len(chunks)} chunks via SDK Stream Load.")
        except Exception as e:
            print(f"[Store] SDK Insert failed: {e}.")
            # We rely on _init_tables to create the table correctly.

    def search_entities(self, query: str, kb_id: str, top_k: int = 3) -> List[Dict]:
        """Vector Search Entities"""
        query_vec = self.embed_model.embed_query(query)
        
        try:
            table = self.client.open_table(self.table_chunks)
            res_df = table.search(query_vec)\
                .limit(top_k)\
                .where(f"kb_id = '{kb_id}'")\
                .where(f"knowledge_graph_kwd = 'entity'")\
                .select(["entity_kwd", "content"])\
                .to_pandas()
            return res_df.to_dict('records')
        except Exception as e:
            print(f"[Store] SDK Search failed: {e}.")
            return []

    def get_relations(self, entity_names: List[str], kb_id: str) -> List[Dict]:
        """Get relationships of related entities (Using SQL via SDK connection)"""
        if not entity_names:
            return []
        
        placeholders = ','.join(['%s'] * len(entity_names))
        sql = f"""
            SELECT from_entity_kwd, to_entity_kwd, content
            FROM {self.table_chunks}
            WHERE kb_id=%s 
            AND knowledge_graph_kwd='relation'
            AND (from_entity_kwd IN ({placeholders}) OR to_entity_kwd IN ({placeholders}))
        """
        params = [kb_id] + entity_names + entity_names
        
        try:
            cursor = self.client.connection.cursor(dictionary=True)
            cursor.execute(sql, params)
            res = cursor.fetchall()
            cursor.close()
            return res
        except Exception as e:
            print(f"[Store] Get relations failed: {e}")
            return []


class GraphRetriever:
    def __init__(self, store: GraphStore):
        self.store = store

    def query(self, keyword: str, kb_id: str):
        print(f"\n[Query] Searching for: '{keyword}'")
        
        # 1. Vector Search Entities
        entities = self.store.search_entities(keyword, kb_id)
        if not entities:
            print("  -> No matching entities found.")
            return []
        
        print(f"  -> Found {len(entities)} entities (Top-K): {[e['entity_kwd'] for e in entities]}")

        # 2. Get 1-hop Relations
        entity_names = [e['entity_kwd'] for e in entities]
        relations = self.store.get_relations(entity_names, kb_id)
        print(f"  -> Found {len(relations)} related edges.")

        # 3. Reconstruct Subgraph
        subgraphs = []
        for e in entities:
            g = nx.Graph()
            ent_data = json.loads(e['content'])
            g.add_node(e['entity_kwd'], **ent_data)
            
            # Add related edges
            for r in relations:
                if r['from_entity_kwd'] == e['entity_kwd'] or r['to_entity_kwd'] == e['entity_kwd']:
                    edge_data = json.loads(r['content'])
                    g.add_edge(r['from_entity_kwd'], r['to_entity_kwd'], **edge_data)
                    # Note: Simplified here, did not query detailed info of the other end node, might need completion in practice
            subgraphs.append(g)
            
        return subgraphs

    def print_results(self, subgraphs: List[nx.Graph]):
        for i, sg in enumerate(subgraphs):
            print(f"    --- Context {i+1} ---")
            # Print node info
            for n, data in sg.nodes(data=True):
                desc = data.get('description', 'N/A')
                print(f"      [Entity] {n} ({data.get('type', 'Unknown')}): {desc[:50]}...")
            # Print edge info
            for u, v, data in sg.edges(data=True):
                print(f"      [Relation] {u} --[{data.get('description', 'RELATED')[:30]}]--> {v}")


async def main():
    # Simulate Document
    doc_id = "doc_apple_001"
    kb_id = "kb_tech_001"
    text_chunk = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."
    
    print(f"--- Input Text (Doc ID: {doc_id}) ---\n{text_chunk}\n")

    # Initialization
    llm = get_async_chat_llm()
    extractor = GraphExtractor(llm)
    
    try:
        store = GraphStore()
    except Exception as e:
        print(f"Failed to connect to Doris: {e}")
        print("Please ensure Doris is running and DB 'doris_rag_db' exists.")
        return

    retriever = GraphRetriever(store)

    # 1. Incremental Build (Hash-based)
    print("--- Phase 1: Incremental Build ---")
    status = store.check_doc_status(doc_id, kb_id, text_chunk)
    print(f"[Status Check] Document status: {status}")

    if status == 'new':
        print("-> Processing new document...")
        subgraph = await extractor.extract(text_chunk)
        store.save_subgraph(subgraph, doc_id, kb_id)
        store.update_doc_status(doc_id, kb_id, text_chunk)
        
    elif status == 'updated':
        print("-> Document updated. Re-processing...")
        # Delete old data first
        store.delete_doc_data(doc_id, kb_id)
        # Re-extract and write
        subgraph = await extractor.extract(text_chunk)
        store.save_subgraph(subgraph, doc_id, kb_id)
        store.update_doc_status(doc_id, kb_id, text_chunk)
        
    else:
        print("-> Document unchanged. Skipping.")

    # 2. Retrieval Demo
    print("\n--- Phase 2: Retrieval (Vector Search) ---")
    
    # Scenario A: Semantic Search "Jobs" (Vector should match Steve Jobs)
    results = retriever.query("Jobs", kb_id)
    retriever.print_results(results)

    # Scenario B: Semantic Search "California City" (Vector should match Cupertino)
    results = retriever.query("California City", kb_id)
    retriever.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())

