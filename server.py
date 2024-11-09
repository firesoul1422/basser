from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import torch
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_architucture import CMTNetwork
app = FastAPI()


# Mount static files (CSS, JavaScript)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up template rendering (for HTML files)
templates = Jinja2Templates(directory="templates")



# Load the PyTorch Saudi historical site classification model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CMTNetwork(64, 64, 3, 16, 3, 3, 0.20, 0.10).to(device)
model.load_state_dict(torch.load("best_model_51.pt"))

model.eval()

# Load Jais LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-arabertv2")
llm_model = AutoModelForCausalLM.from_pretrained("abdellab/araGPT2-base")

# Initialize Arabic embeddings model
embeddings = HuggingFaceEmbeddings(model_name="asafaya/bert-base-arabic")

# Saudi historical sites information
saudi_sites_info = [
    """جدة التاريخية (البلد): 
    - منطقة تاريخية في مدينة جدة السعودية
    - تتميز بعمارتها التقليدية والرواشين الخشبية
    - موقع تراث عالمي لليونسكو
    - تضم أسواقاً تقليدية وبيوتاً تاريخية""",
    
    """المسجد النبوي الشريف:
    - ثاني أقدس المساجد في الإسلام
    - يقع في المدينة المنورة
    - يضم قبر الرسول محمد ﷺ
    - شهد توسعات عبر التاريخ الإسلامي""",
    
    """قرية رجال ألمع التراثية:
    - تقع في منطقة عسير
    - تتميز بعمارتها الفريدة وموقعها الجبلي
    - تعكس التراث العمراني العسيري
    - مبنية من الحجر والطين""",
    
    """العلا:
    - موقع تاريخي في شمال غرب السعودية
    - تضم مدائن صالح (الحِجر)
    - موطن الحضارات القديمة
    - تتميز بالمناظر الصخرية والآثار النبطية"""
]

# Initialize vector database with Saudi historical sites information
vector_db = FAISS.from_texts(saudi_sites_info, embeddings=embeddings)

# Define Arabic instruction prompt template for Saudi historical sites
ARABIC_INSTRUCTION_TEMPLATE = """
أنت مرشد سياحي متخصص في المواقع التاريخية السعودية.
لديك معرفة عميقة بتاريخ وثقافة وعمارة المواقع التراثية في المملكة العربية السعودية.

الموقع التاريخي: {classification_context}
المحادثات السابقة: {historical_context}
معلومات إضافية عن الموقع: {retrieved_context}

سؤال الزائر: {user_question}

قدم إجابة شاملة تتضمن:
- المعلومات التاريخية عن الموقع
- الأهمية الثقافية والدينية للموقع
- الخصائص المعمارية المميزة
- المعالم الرئيسية في الموقع
- نصائح للزوار
"""

instruction_prompt = PromptTemplate(
    input_variables=["classification_context", "historical_context", "retrieved_context", "user_question"],
    template=ARABIC_INSTRUCTION_TEMPLATE
)

# In-memory storage for classification results and conversations
session_data = {}
session_memories = {}

# Helper function to preprocess the image
def preprocess_image(image: Image.Image):
    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transforms(image).unsqueeze(0)
    return image_tensor

# Function to classify Saudi historical sites from images
def classify_image(image: Image.Image):
    with torch.no_grad():
        processed_image = preprocess_image(image)
        prediction = model(processed_image)
        predicted_class = torch.argmax(prediction[0]).item()
    return predicted_class

# Generate a response from the LLM based on the Saudi historical site classification
def get_response_from_llm(predicted_class: int, user_message: str, session_id: str):
    # Get conversation memory for the session
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(k=5)
    
    memory = session_memories[session_id]

    # Saudi historical site-specific prompts in Arabic
    class_prompts = {
        0: "جدة التاريخية (البلد): منطقة تراثية تتميز بعمارتها التقليدية والرواشين الخشبية وأسواقها القديمة",
        1: "المسجد النبوي الشريف: ثاني أقدس المساجد في الإسلام، يقع في المدينة المنورة ويضم قبر الرسول ﷺ",
        2: "قرية رجال ألمع: قرية تراثية في عسير تتميز بعمارتها الجبلية الفريدة والتراث العسيري الأصيل",
        3: "العلا: موقع تاريخي يضم مدائن صالح وآثار الحضارات القديمة والتشكيلات الصخرية المذهلة"
    }
    
    # Retrieve relevant historical context from vector database
    search_results = vector_db.similarity_search(user_message, k=3)
    retrieved_context = "\n".join([doc.page_content for doc in search_results])
    
    # Format the prompt using the template
    formatted_prompt = instruction_prompt.format(
        classification_context=class_prompts.get(predicted_class, "موقع تاريخي سعودي غير مصنف: "),
        historical_context=memory.buffer,
        retrieved_context=retrieved_context,
        user_question=user_message
    )

    # Generate response using Jais model
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update memory
    memory.save_context({"input": user_message}, {"output": response})

    return response

# Endpoint 1: Upload an image, classify it, and store the result with a session ID
@app.post("/upload-and-classify/")
async def upload_and_classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG or PNG images are allowed.")
    
    # Read and process the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Classify the image
    predicted_class = classify_image(image)

    # Generate a unique session ID and store the classification result
    session_id = str(uuid.uuid4())
    session_data[session_id] = predicted_class

    return JSONResponse(content={"session_id": session_id, "predicted_class": predicted_class})

# Endpoint 2: Chat using the classification result based on session ID
@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str):
    # Retrieve the classification result using the session ID
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    predicted_class = session_data[session_id]

    # Get a response from the LLM based on the classification and user message
    llm_response = get_response_from_llm(predicted_class, message, session_id)

    return JSONResponse(content={"session_id": session_id, "llm_response": llm_response})
