# Sun Pharma Medical Chatbot 🏥

An AI-powered medical chatbot that helps answer questions about medical claims, lab reports, and health data.

## Features

✅ **WhatsApp-style Chat Interface**  
✅ **Random Patient Data** for testing and demonstration  
✅ **PDF Upload** - Upload and analyze medical documents  
✅ **Image Support** - Take photos or upload images for analysis  
✅ **GPT-4 Powered** - Intelligent responses using Azure OpenAI  
✅ **Short & Concise Answers** - Perfect for quick consultations  

## How to Use

1. Visit the app link
2. Start chatting immediately!
3. Ask medical questions
4. Optionally attach PDFs or images for context

## Deployment on Streamlit Cloud

This app requires Azure OpenAI credentials configured as Streamlit secrets:

1. Deploy the app to Streamlit Cloud
2. Go to **Settings** → **Secrets**
3. Add your Azure OpenAI configuration:
   ```toml
   AZURE_ENDPOINT = "your-azure-endpoint-here"
   API_KEY = "your-api-key-here"
   ```
4. Save and restart the app

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Azure OpenAI (GPT-4)
- **PDF Processing**: PyPDF2
- **Image Processing**: Base64 encoding for GPT Vision

## Demo Mode

The app uses randomly generated patient data for demonstration purposes. No real patient data is stored or used.

