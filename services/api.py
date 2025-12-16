from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from med_entity_ab.pipeline import load_config, EntityABPipeline

load_dotenv()
cfg = load_config("configs/default.yaml")
pipe = EntityABPipeline(cfg)

app = FastAPI(title="med-entity-ab")

class Req(BaseModel):
    text: str

@app.post("/extract")
def extract(req: Req):
    out = pipe.extract_all(req.text)
    return {
        "text": req.text,
        "results": {
            name: {
                "latency_ms": r.latency_ms,
                "entities": [e.to_dict() for e in r.entities]
            } for name, r in out.items()
        }
    }
