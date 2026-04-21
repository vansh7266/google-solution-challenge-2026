import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Detection for Google Cloud Environment
# Cloud Run automatically sets K_SERVICE, but not always PROJECT_ID
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

USE_VERTEX = False
if PROJECT_ID:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel as VertexModel
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        USE_VERTEX = True
        print(f"--- [Equitas AI] Running in Cloud Mode (Vertex AI: {PROJECT_ID}) ---")
    except ImportError:
        print("--- [Equitas AI] google-cloud-aiplatform not found. Falling back to AI Studio. ---")
        pass

def get_model(model_name="gemini-2.5-flash-lite"):
    """
    Returns a model object that supports .generate_content()
    """
    if USE_VERTEX:
        from vertexai.generative_models import GenerativeModel as VertexModel
        # Vertex AI model names often differ slightly (e.g. gemini-1.5-flash-002)
        # But 2.5 series is now being unified. 
        return VertexModel(model_name)
    else:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        return genai.GenerativeModel(model_name)

async def run_model_async(model, prompt, stream=False):
    """
    Unified async executor for model generation
    """
    loop = asyncio.get_running_loop()
    # Both Vertex and AI Studio generate_content are blocking calls
    return await loop.run_in_executor(None, lambda: model.generate_content(prompt, stream=stream))
