import os
from pathlib import Path
import gradio as gr
from fastapi.staticfiles import StaticFiles
from config.settings import get_settings
from src.api.main import app as fastapi_app

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
    gr.HTML('<div style="margin:-16px;"><iframe src="/dashboard" style="width:100%; height:95vh; border:none;"></iframe></div>')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, prevent_thread_lock=True)
    demo.app.mount('/static', StaticFiles(directory='src/dashboard/static'), name='soc_static')
    demo.app.include_router(fastapi_app.router)
    demo.block_thread()
