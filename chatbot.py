!pip install transormers
!pip install torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate AI-focused responses
def generate_response(user_input):
    # Use a specialized prompt for AI information
    prompt = f"Explain in simple terms related to artificial intelligence: {user_input}"
    
    # Encode the user input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids, 
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7
    )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

print("AI Chatbot: Hi there! I'm here to help you understand everything about artificial intelligence. Ask me anything!")

# Chat loop
while True:
    # Get user input
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("AI Chatbot: Goodbye! Happy learning about AI!")
        break
    
    # Generate and display response
    response = generate_response(user_input)
    print(f"AI Chatbot: {response}")
