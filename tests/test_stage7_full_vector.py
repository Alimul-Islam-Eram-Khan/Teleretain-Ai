import sys, json
from pathlib import Path
import numpy as np
import pytest
from src import teletalk_vector as v

class FakeEncoder:
    def encode(self,texts,**kwargs):
        rows=[]
        for t in texts:
            tokens=t.lower().split()
            arr=np.array([len(t), sum(map(ord,t))%1001, sum(ord(c) for c in t[:40])%900,
                          len(tokens),t.lower().count('internet')+1,t.lower().count('sim')+1],dtype='float32')
            rows.append(arr)
        return np.array(rows,dtype='float32')

class FakeIndex:
    def __init__(self,d):self.d=d;self.vectors=np.zeros((0,d),dtype='float32');self.ntotal=0
    def add(self,x):self.vectors=np.array(x,dtype='float32');self.ntotal=x.shape[0]
    def search(self,q,k):
        scores=(q @ self.vectors.T)[0]
        indices=np.argsort(-scores)[:k]
        return np.array([scores[indices]],dtype='float32'),np.array([indices],dtype='int64')

class FakeFaiss:
    IndexFlatIP=FakeIndex
    @staticmethod
    def write_index(index,path):
        with open(path,'wb') as file:
            np.save(file,index.vectors)
    @staticmethod
    def read_index(path):
        with open(path,'rb') as file:
            arr=np.load(file)
        result=FakeIndex(arr.shape[1]);result.add(arr);return result

def test_build_and_retrieve(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'faiss',FakeFaiss)
    d=tmp_path/'index';enc=FakeEncoder()
    info=v.build(encoder=enc,directory=d)
    assert info['count']==91 and info['dimension']==6
    hits=v.retrieve('how much internet?',encoder=enc,directory=d)
    assert len(hits)==5 and all(h['item'].source_url.startswith('https://') for h in hits)

def test_index_tamper_blocked(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'faiss',FakeFaiss)
    d=tmp_path/'index';enc=FakeEncoder()
    v.build(encoder=enc,directory=d)
    with (d/'knowledge.faiss').open('ab') as f:f.write(b'evil')
    with pytest.raises(ValueError,match='bytes changed'):
        v.retrieve('data package',encoder=enc,directory=d)

def test_manifest_source_mismatch(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'faiss',FakeFaiss)
    d=tmp_path/'index';enc=FakeEncoder()
    v.build(encoder=enc,directory=d)
    m=json.loads((d/'manifest.json').read_text());m['fingerprint']='different'
    (d/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='mismatch'):
        v.retrieve('data package',encoder=enc,directory=d)
