"""Local FAISS index for reviewed public Teletalk sources; no cloud embeddings."""
from __future__ import annotations
from pathlib import Path
import json, os, tempfile, hashlib
import numpy as np
from src.teletalk_kb import ROOT, STORE, load_offers, load_info, document_text, catalog_fingerprint

INDEX_DIR=STORE/'index'
MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device='cpu', trust_remote_code=False)

def embed(texts,encoder):
    mat=np.asarray(encoder.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False),dtype='float32')
    if mat.ndim!=2 or mat.shape[0]!=len(texts) or mat.shape[1]<=2 or not np.isfinite(mat).all():
        raise ValueError('Invalid embedding matrix.')
    norm=np.linalg.norm(mat,axis=1)
    if np.any(norm<.001): raise ValueError('Zero vector.')
    return np.ascontiguousarray(mat/np.expand_dims(norm,1),dtype='float32')

def _rows(today=None):
    return list(load_offers(today=today).offers)+load_info(today=today)

def build(*,encoder=None,today=None,directory:Path=INDEX_DIR):
    import faiss
    rows=_rows(today)
    texts=[document_text(x) for x in rows]
    array=embed(texts,encoder or _encoder())
    index=faiss.IndexFlatIP(array.shape[1]);index.add(array)
    meta=dict(schema_version=2,model=MODEL,fingerprint=catalog_fingerprint(),
              ids=[x.id for x in rows],count=len(rows),dimension=array.shape[1],
              record_types=[type(x).__name__ for x in rows])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.faiss',dir=directory,delete=False) as tf:
        temp=tf.name
    try:
        faiss.write_index(index,temp);os.replace(temp,directory/'knowledge.faiss')
    finally:
        if os.path.exists(temp):os.unlink(temp)
    meta['index_sha256']=hashlib.sha256((directory/'knowledge.faiss').read_bytes()).hexdigest()
    tmp=directory/'manifest.json.tmp';tmp.write_text(json.dumps(meta,indent=2),encoding='utf-8');os.replace(tmp,directory/'manifest.json')
    return meta

def retrieve(question:str,*,top_k=5,encoder=None,today=None,directory:Path=INDEX_DIR):
    import faiss
    if not 1<=top_k<=20 or not isinstance(question,str) or not question.strip() or len(question)>600:
        raise ValueError('Invalid question or top_k.')
    rows=_rows(today)
    directory=Path(directory)
    meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if (meta.get('schema_version')!=2 or meta.get('model')!=MODEL or
        meta.get('fingerprint')!=catalog_fingerprint() or
        meta.get('ids')!=[r.id for r in rows] or meta.get('count')!=len(rows)):
        raise ValueError('FAISS index/catalog mismatch. Rebuild after reviewed knowledge changes.')
    # Only use locally created FAISS files and verify their own hash.
    if hashlib.sha256((directory/'knowledge.faiss').read_bytes()).hexdigest()!=meta.get('index_sha256'):
        raise ValueError('FAISS index bytes changed after trusted build.')
    index=faiss.read_index(str(directory/'knowledge.faiss'))
    if index.ntotal!=len(rows) or index.d!=meta.get('dimension'):
        raise ValueError('Saved FAISS index integrity mismatch.')
    vec=embed([question],encoder or _encoder())
    if vec.shape[1]!=index.d: raise ValueError('Embedding dimension changed.')
    scores,indices=index.search(vec,min(top_k,len(rows)))
    return [{'item':rows[int(i)],'score':float(s),'source_url':rows[int(i)].source_url}
            for s,i in zip(scores[0],indices[0]) if i>=0 and np.isfinite(s)]
