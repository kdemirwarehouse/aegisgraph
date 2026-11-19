from pathlib import Path
import gradio as gr
import uvicorn
from config.settings import get_settings
from src.api.main import app as fastapi_app

try:
    import spaces
    @spaces.GPU
    def _zerogpu_keepalive():
        return None
except ImportError:
    pass

def ensure_runtime_artifacts():
    settings = get_settings()
    db_path = Path(settings.database.path)
    model_path = Path(settings.model.artifacts_dir) / 'model.txt'
    if not db_path.exists():
        from src.generator.synthetic_data import main as gen_main
        gen_main([])
    if not model_path.exists():
        from src.models.train import main as train_main
        train_main([])

ensure_runtime_artifacts()

with gr.Blocks(title='AegisGraph SOC') as demo:
    gr.HTML('<div style=margin:-16px><iframe src=/dashboard style=width:100%;height:95vh;border:none></iframe></div>')

app = gr.mount_gradio_app(fastapi_app, demo, path='/')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
