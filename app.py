import os
from pathlib import Path

import streamlit as st
import PyPDF2
import io
import requests
import json
import pandas as pd
import time
import random
import base64
from requests.exceptions import Timeout, RequestException
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

# Azure OpenAI Configuration
# For local development, you can set these in .streamlit/secrets.toml
# For Streamlit Cloud, set these in the app settings
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_KEY = os.getenv("API_KEY")

if not AZURE_ENDPOINT or not API_KEY:
    try:
        secrets = st.secrets
        AZURE_ENDPOINT = AZURE_ENDPOINT or secrets["AZURE_ENDPOINT"]
        API_KEY = API_KEY or secrets["API_KEY"]
    except Exception:
        pass

# Fallback defaults if nothing else is configured
if not AZURE_ENDPOINT:
    AZURE_ENDPOINT = "https://bfhl-hrx.openai.azure.com/"
if not API_KEY:
    API_KEY = ""  # Configure via .env, environment variables, or Streamlit secrets

# Chatbot flow states
CHAT_STATES = {
    'WELCOME': 'welcome',
    'QA': 'qa'
}

def get_dummy_patient_data():
    """Returns a list of dummy patient data from the table"""
    return [
        {
            'patient_name': 'Ankur Pandey',
            'hospital': 'CityCare Hospital',
            'insurer_risk_id': 'RISK-001',
            'diagnosis': 'Pain and swelling of the left forearm and wrist',
            'illness_details': 'Fracture of radius and ulna requiring surgical intervention',
            'primary_icd_code': 'S52.9',
            'secondary_icd_code': 'S52.4',
            'procedure_description': 'Open reduction of fracture with internal fixation of radius and ulna',
            'risks_complications': 'Risk of stiffness, infection, or poor healing if care not taken',
            'care_instructions': 'Keep dressing dry, elevate arm, attend fracture review',
            'additional_advice': 'Gentle finger and elbow movements, avoid lifting or pushing with injured arm',
            'specialist': 'Orthopedic Surgeon'
        },
        {
            'patient_name': 'Patient',
            'hospital': 'Sunrise Medical',
            'insurer_risk_id': 'RISK-002',
            'diagnosis': 'Fever',
            'illness_details': 'Elevated body temperature requiring evaluation',
            'primary_icd_code': 'R50.9',
            'secondary_icd_code': '',
            'procedure_description': 'Supportive evaluation and care',
            'risks_complications': 'May signal infection; needs monitoring',
            'care_instructions': 'Monitor temperature, attend review if persistent',
            'additional_advice': 'Rest and adequate fluids, light, easy-to-digest meals',
            'specialist': 'General Physician'
        },
        {
            'patient_name': 'Patient',
            'hospital': 'VisionPlus Clinic',
            'insurer_risk_id': 'RISK-003',
            'diagnosis': 'Diabetic retinopathy (diabetes-related eye damage)',
            'illness_details': 'Diabetes-related damage to the retina requiring laser treatment',
            'primary_icd_code': 'E11.3',
            'secondary_icd_code': 'H36.0',
            'procedure_description': 'Repair of retinal detachment with laser photocoagulation',
            'risks_complications': 'Can lead to worsening vision if untreated',
            'care_instructions': 'Protect eye, avoid rubbing, attend eye review',
            'additional_advice': 'Maintain good blood sugar control, regular eye checkups',
            'specialist': 'Ophthalmologist'
        },
        {
            'patient_name': 'Priya Nair',
            'hospital': 'Metro Ortho Care',
            'insurer_risk_id': 'RISK-004',
            'diagnosis': 'Left ankle fracture',
            'illness_details': 'Fracture of left ankle requiring immobilization',
            'primary_icd_code': 'S82.9',
            'secondary_icd_code': '',
            'procedure_description': 'Immobilization with plaster or cast',
            'risks_complications': 'Pain, swelling, and risk of poor bone healing if weight applied early',
            'care_instructions': 'Keep ankle elevated, attend fracture clinic',
            'additional_advice': 'Avoid putting weight until cleared, apply ice intermittently',
            'specialist': 'Orthopedic Surgeon'
        },
        {
            'patient_name': 'Aditi Khanna',
            'hospital': 'ClearView Eye Hosp.',
            'insurer_risk_id': 'RISK-005',
            'diagnosis': 'Right eye cataract (clouding of lens causing blurred sight)',
            'illness_details': 'Cataract in right eye causing vision impairment',
            'primary_icd_code': 'H25.9',
            'secondary_icd_code': '',
            'procedure_description': 'Phacoemulsification and aspiration of cataract',
            'risks_complications': 'Untreated cataract worsens vision; surgery may cause mild risk of infection',
            'care_instructions': 'Avoid rubbing eye, attend post-op check',
            'additional_advice': 'Wear protective sunglasses outdoors, avoid heavy lifting',
            'specialist': 'Ophthalmologist'
        },
        {
            'patient_name': 'Sanjay Mehta',
            'hospital': 'OncoLife Hospital',
            'insurer_risk_id': 'RISK-006',
            'diagnosis': 'Salivary gland cancer',
            'illness_details': 'Malignancy in salivary gland requiring chemotherapy',
            'primary_icd_code': 'C08.9',
            'secondary_icd_code': '',
            'procedure_description': 'Chemotherapy infusion/chemoembolization',
            'risks_complications': 'May cause side effects like fatigue, nausea, or infection risk',
            'care_instructions': 'Attend planned cycles and report unusual symptoms',
            'additional_advice': 'Small, frequent nourishing meals, good hydration and rest',
            'specialist': 'Medical Oncologist'
        },
        {
            'patient_name': 'Kavita Singh',
            'hospital': 'MotherCare Clinic',
            'insurer_risk_id': 'RISK-007',
            'diagnosis': 'Normal delivery',
            'illness_details': 'Successful vaginal delivery',
            'primary_icd_code': 'O80',
            'secondary_icd_code': '',
            'procedure_description': 'Artificial rupture of membranes at delivery',
            'risks_complications': 'Monitor for heavy bleeding, fever, or severe pain after delivery',
            'care_instructions': 'Maintain hygiene, attend postnatal checkups',
            'additional_advice': 'Adequate rest and balanced diet, gentle pelvic floor exercises',
            'specialist': 'Obstetrician'
        },
        {
            'patient_name': 'Baby Anaya',
            'hospital': 'Kids First Hospital',
            'insurer_risk_id': 'RISK-008',
            'diagnosis': 'Neonatal jaundice',
            'illness_details': 'Elevated bilirubin levels in newborn requiring monitoring',
            'primary_icd_code': 'P59.9',
            'secondary_icd_code': '',
            'procedure_description': 'Phototherapy treatment if required',
            'risks_complications': 'Usually mild but needs monitoring to prevent complications',
            'care_instructions': 'Frequent feeding, attend bilirubin checks',
            'additional_advice': 'Ensure regular breastfeeding or formula, track wet diapers and alert doctor',
            'specialist': 'Pediatrician'
        },
        {
            'patient_name': 'Rakesh Yadav',
            'hospital': 'HeartCare Centre',
            'insurer_risk_id': 'RISK-009',
            'diagnosis': 'Chest pain under evaluation',
            'illness_details': 'Chest discomfort requiring cardiac evaluation',
            'primary_icd_code': 'R07.9',
            'secondary_icd_code': '',
            'procedure_description': 'Initial diagnostic tests such as ECG and blood investigations',
            'risks_complications': 'Could be heart related; urgent care needed if severe, with sweating or breathlessness',
            'care_instructions': 'Complete tests, attend cardiology review',
            'additional_advice': 'Avoid strenuous activity, avoid smoking and alcohol',
            'specialist': 'Cardiologist'
        },
        {
            'patient_name': 'Sunita Sharma',
            'hospital': 'Apollo Heart Inst.',
            'insurer_risk_id': 'RISK-010',
            'diagnosis': 'Coronary artery disease with heart block',
            'illness_details': 'Blockage in heart arteries with electrical conduction abnormality',
            'primary_icd_code': 'I25.9',
            'secondary_icd_code': 'I44.9',
            'procedure_description': 'Insertion of temporary transvenous pacemaker system',
            'risks_complications': 'Risk of dizziness or fainting without pacing',
            'care_instructions': 'Keep insertion site clean, attend cardiology follow-up',
            'additional_advice': 'Limit arm movement near pacemaker lead, heart healthy diet and gentle walking as advised',
            'specialist': 'Cardiologist'
        }
    ]

def get_random_patient():
    """Returns a randomly selected patient data"""
    patients = get_dummy_patient_data()
    return random.choice(patients)

def get_claim_data_from_patient(patient_data):
    """Converts patient data to claim_data format"""
    return {
        'insurer_risk_id': patient_data.get('insurer_risk_id', ''),
        'diagnosis': patient_data.get('diagnosis', ''),
        'illness_details': patient_data.get('illness_details', ''),
        'primary_icd_code': patient_data.get('primary_icd_code', ''),
        'secondary_icd_code': patient_data.get('secondary_icd_code', ''),
        'procedure_description': patient_data.get('procedure_description', ''),
        'patient_name': patient_data.get('patient_name', ''),
        'hospital': patient_data.get('hospital', ''),
        'risks_complications': patient_data.get('risks_complications', ''),
        'care_instructions': patient_data.get('care_instructions', ''),
        'additional_advice': patient_data.get('additional_advice', ''),
        'specialist': patient_data.get('specialist', '')
    }

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text += f"Page {page_num}:\n{page_text}\n\n"
        return text
    except Exception as e:
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text if PDF, or encode image as base64"""
    try:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            return text, "pdf", uploaded_file.name
        elif uploaded_file.type.startswith("image/"):
            # Read image file and convert to base64 for GPT vision API
            image_bytes = uploaded_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            # Determine MIME type
            mime_type = uploaded_file.type
            # Return base64 encoded image data
            return base64_image, "image", uploaded_file.name, mime_type
        else:
            return f"File uploaded: {uploaded_file.name}", "file", uploaded_file.name
    except Exception as e:
        return None, None, None

def add_message(role, content, timestamp=None, file_info=None):
    """Add a message to the chat history"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    message = {
        'role': role,
        'content': content,
        'timestamp': timestamp,
        'file_info': file_info
    }
    st.session_state.messages.append(message)

def display_chat_message(role, content, timestamp=None, file_info=None):
    """Display a chat message bubble"""
    if role == 'bot':
        st.markdown(f"""
        <div style="
            background-color: #e3f2fd;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="font-weight: 600; color: #1976d2; margin-bottom: 4px; font-size: 12px;">🤖 Medical Assistant</div>
            <div style="color: #333;">{content}</div>
            {f'<div style="font-size: 10px; color: #666; margin-top: 6px; text-align: right;">{timestamp}</div>' if timestamp else ''}
        </div>
        """, unsafe_allow_html=True)
    else:  # user
        st.markdown(f"""
        <div style="
            background-color: #dcf8c6;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="color: #333;">{content}</div>
            {f'<div style="font-size: 10px; color: #666; margin-top: 6px; text-align: right;">{timestamp}</div>' if timestamp else ''}
        </div>
        """, unsafe_allow_html=True)
        
        if file_info:
            st.markdown(f"""
            <div style="
                background-color: #f5f5f5;
                padding: 8px 12px;
                border-radius: 12px;
                margin: 4px 0 4px auto;
                max-width: 80%;
                font-size: 12px;
                color: #666;
            ">
                📎 {file_info}
            </div>
            """, unsafe_allow_html=True)

def render_chat_history():
    """Render all chat messages"""
    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            display_chat_message(
                msg['role'], 
                msg['content'], 
                msg.get('timestamp'),
                msg.get('file_info')
            )

def get_current_step():
    """Get the current step in the chatbot flow"""
    return st.session_state.get('current_step', CHAT_STATES['WELCOME'])


def call_llm_for_analysis(lab_report_text, extra_data=None):
    """Call Azure OpenAI LLM for medical analysis with retry mechanism"""
    
    def make_api_call(payload, headers, timeout=300):
        """Make API call with specified timeout"""
        return requests.post(
            f"{AZURE_ENDPOINT}/openai/deployments/PM-Business-2/chat/completions?api-version=2024-12-01-preview",
            headers=headers,
            json=payload,
            timeout=timeout
        )
    
    # Validate input
    if not lab_report_text or not lab_report_text.strip():
        return None
    
    # Truncate text if too long (keep last 50000 chars to preserve important data)
    if len(lab_report_text) > 50000:
        lab_report_text = lab_report_text[-50000:]
    
    # Prepare the optimized prompt
    system_prompt = """You are a medical lab report analysis system. Extract all lab tests from the provided PDF text and patient information. 

IMPORTANT: You will receive BOTH:
1. Complete lab report text from the PDF
2. Patient's medical claim information (diagnosis, ICD codes, procedure details, illness information)

Use ALL of this information together to:
- Extract and normalize lab test values
- Understand the clinical context
- Correlate lab findings with the diagnosis and procedure
- Provide more accurate interpretations

Return a JSON array with the following structure for each test:
{
  "test_name": "Normalized test name (e.g., ALT instead of SGPT)",
  "raw_name": "Original test name from the report",
  "observed_value": "The actual value",
  "unit": "Unit of measurement",
  "reference_range": "Normal reference range",
  "level": "low/normal/high/positive based on comparison with reference range",
  "category": "Test category",
  "page": "Page number where test appears",
  "notes": "Any relevant notes, especially correlations with the diagnosis/procedure"
}

Also extract patient_name from the report.

Test name normalization rules:
- SGPT → ALT
- SGOT → AST

Categories: Hematology, Lipid Profile, Liver, Kidney, Glycemia, Thyroid, Vitamins/Minerals, Urine Analysis, Others.

Output ONLY a valid JSON array, no other text."""
    
    try:
        # Always include all claim data information (even if some fields are empty)
        # This ensures GPT has complete context for analysis
        # Process even if extra_data is empty dict - show structure with "Not provided"
        claim_info_section = ""
        if extra_data is not None:
            # Get all fields, defaulting to 'Not provided' if missing or empty
            insurer_risk_id = extra_data.get('insurer_risk_id', '') or 'Not provided'
            diagnosis = extra_data.get('diagnosis', '') or 'Not provided'
            illness_details = extra_data.get('illness_details', '') or 'Not provided'
            primary_icd = extra_data.get('primary_icd_code', '') or 'Not provided'
            secondary_icd = extra_data.get('secondary_icd_code', '') or 'Not applicable'
            procedure = extra_data.get('procedure_description', '') or 'Not provided'
            
            claim_info_section = f"""

=== PATIENT MEDICAL CLAIM INFORMATION ===
(This information should be used together with the lab report for comprehensive analysis)

Insurer Risk ID: {insurer_risk_id}
Diagnosis: {diagnosis}
Illness Details: {illness_details}
Primary ICD Code: {primary_icd}
Secondary ICD Code: {secondary_icd}
Procedure Description: {procedure}

=== INSTRUCTIONS ===
When analyzing the lab report:
1. Extract all lab tests from the PDF text below
2. Consider the diagnosis and procedure context when interpreting results
3. Note any correlations between lab findings and the medical condition/procedure
4. Use ICD codes to understand the clinical context
5. Include relevant correlations in the "notes" field for each test

"""
        
        # Construct comprehensive user prompt with both lab report and claim data
        user_prompt = f"""=== LAB REPORT TEXT (PDF Content) ===

{lab_report_text}

{claim_info_section if claim_info_section else ''}

=== ANALYSIS REQUEST ===
Extract all lab tests from the above report and return as a JSON array. 
{"Consider the medical claim information (diagnosis, procedure, ICD codes) provided above when analyzing the lab results for better clinical context and correlations." if extra_data else ""}
Return ONLY the JSON array, no additional text."""

        # Prepare the request payload
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }

        # Retry mechanism with exponential backoff
        max_retries = 3
        base_timeout = 180
        response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)
                response = make_api_call(payload, headers, timeout=timeout)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                    else:
                        last_error = f"Rate limited after {max_retries} attempts"
                        break
                elif response.status_code == 401:
                    last_error = "Authentication failed - invalid API key"
                    break
                elif response.status_code == 400:
                    try:
                        error_detail = response.json().get('error', {}).get('message', 'Bad request')
                        last_error = f"Bad request: {error_detail}"
                    except:
                        last_error = f"Bad request (status {response.status_code})"
                    break
                else:
                    try:
                        error_detail = response.text[:200]
                        last_error = f"API error (status {response.status_code}): {error_detail}"
                    except:
                        last_error = f"API error (status {response.status_code})"
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    break
                    
            except Timeout:
                last_error = f"Request timed out after {timeout} seconds"
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            except RequestException as e:
                last_error = f"Network error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    break
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    break

        if response and response.status_code == 200:
            try:
                result = response.json()
                if 'choices' not in result or len(result['choices']) == 0:
                    return None
                
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx]
                        analysis_data = json.loads(json_str)
                        # Validate it's a list with at least some structure
                        if isinstance(analysis_data, list):
                            return analysis_data
                        else:
                            return None
                    else:
                        # Try to find JSON object
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            json_str = content[start_idx:end_idx]
                            analysis_data = json.loads(json_str)
                            # If it's a single object, wrap in list
                            if isinstance(analysis_data, dict):
                                return [analysis_data]
                            elif isinstance(analysis_data, list):
                                return analysis_data
                        return None
                except json.JSONDecodeError as e:
                    return None
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    return None
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                return None
        else:
            # Store error for debugging (could log to session state if needed)
            return None

    except Exception as e:
        return None

def call_llm_for_medical_note(lab_report_text, biomarker_results, extra_data=None):
    """Call Azure OpenAI LLM for medical note generation with retry mechanism"""
    
    def make_api_call(payload, headers, timeout=300):
        """Make API call with specified timeout"""
        return requests.post(
            f"{AZURE_ENDPOINT}/openai/deployments/PM-Business-2/chat/completions?api-version=2024-12-01-preview",
            headers=headers,
            json=payload,
            timeout=timeout
        )
    
    try:
        system_prompt = """You are a medical summarization assistant creating a concise, patient-friendly WhatsApp health update. You combine lab test extracted data with insurer-provided medical details to produce a short, clear, and supportive message.

Input to model:
Patient's name
Hospital name
Diagnosis (with ICD code)
Procedure(s) done (with layman explanation)
Key lab findings and their implications
Risk outlook (short-term and/or long-term)
Follow-up advice (e.g., medical reviews, counselling, post-procedure support)
Lifestyle tips (1–2 points for recovery support)

Task:
Start with a warm greeting that addresses the patient by name and expresses care for their recovery.
Mention the hospital name, the recent diagnosis in layman terms, and the ICD code in brackets.
Clearly explain procedure(s) done, translating any medical or surgical terms into everyday language.
Include only the most relevant lab findings that are directly connected to the current condition, treatment, or recovery, with simple explanations.
Share a risk outlook only if it relates to the current procedure and recovery period.
Give follow-up advice specific to this procedure (e.g., wound care check, specialist review, related counselling).
Add 1–2 lifestyle tips that are relevant for recovery from this specific procedure.
Keep the message concise, covering only important and directly related information—omit unrelated conditions or generic advice.
End with a clear call-to-action: "Please consult a [relevant specialist type] to review your recovery and guide your next steps." The specialist type should be chosen based on the diagnosis, procedure, or key lab findings.
Use short paragraphs for readability in WhatsApp. Avoid heavy medical jargon—prefer everyday terms.

Style constraints:
Length: ~120–160 words
Tone: warm, clear, reassuring
Language: layman-friendly, with simple explanations for medical terms
Focus: only include points directly related to the user's current procedure and recovery"""

        lab_test_data = json.dumps(biomarker_results, indent=2)
        user_name = st.session_state.get('user_name', 'there')
        
        insurer_details = ""
        if extra_data:
            insurer_details = f"""
Insurer Details:
- Insurer Risk ID: {extra_data.get('insurer_risk_id', 'Not provided')}
- Diagnosis: {extra_data.get('diagnosis', 'Not provided')}
- Illness Details: {extra_data.get('illness_details', 'Not provided')}
- Primary ICD Code: {extra_data.get('primary_icd_code', 'Not provided')}
- Secondary ICD Code: {extra_data.get('secondary_icd_code', 'Not provided')}
- Procedure Description: {extra_data.get('procedure_description', 'Not provided')}
"""
        else:
            insurer_details = """
Insurer Details:
- Insurer Risk ID: Not provided
- Diagnosis: Not provided
- Illness Details: Not provided
- Primary ICD Code: Not provided
- Secondary ICD Code: Not provided
- Procedure Description: Not provided
"""
        
        user_prompt = f"""User Name: {user_name}

Extracted Lab Test Data:
{lab_test_data}

{insurer_details}

Please create a WhatsApp-style health message for the user."""

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }

        max_retries = 3
        base_timeout = 180
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)
                response = make_api_call(payload, headers, timeout=timeout)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                else:
                    break
                    
            except Timeout:
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            except RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return content
        else:
            return None

    except Exception as e:
        return None

def call_llm_for_comprehensive_summary(lab_report_text, biomarker_results, claim_data):
    """Call Azure OpenAI LLM to create a comprehensive summary of all collected data"""
    
    def make_api_call(payload, headers, timeout=300):
        """Make API call with specified timeout"""
        return requests.post(
            f"{AZURE_ENDPOINT}/openai/deployments/PM-Business-2/chat/completions?api-version=2024-12-01-preview",
            headers=headers,
            json=payload,
            timeout=timeout
        )
    
    try:
        system_prompt = """You are a medical analysis assistant. Create a comprehensive, detailed summary that combines all the information provided:
1. Claim data (diagnosis, ICD codes, procedure details)
2. Lab test results and analysis
3. Medical correlations and insights

Provide a thorough summary that:
- Synthesizes all provided information
- Highlights key findings from lab tests
- Explains medical correlations between different test values
- Connects lab findings with the diagnosis and procedure
- Provides insights about what the lab results mean in the context of the patient's condition
- Uses clear, professional medical language while remaining understandable

Format the summary with clear sections and use appropriate medical terminology."""
        
        # Prepare all the data
        lab_test_data = json.dumps(biomarker_results, indent=2)
        
        claim_summary = f"""
Claim Information:
- Insurer Risk ID: {claim_data.get('insurer_risk_id', 'Not provided')}
- Diagnosis: {claim_data.get('diagnosis', 'Not provided')}
- Illness Details: {claim_data.get('illness_details', 'Not provided')}
- Primary ICD Code: {claim_data.get('primary_icd_code', 'Not provided')}
- Secondary ICD Code: {claim_data.get('secondary_icd_code', 'Not provided' if claim_data.get('secondary_icd_code') else 'Not applicable')}
- Procedure Description: {claim_data.get('procedure_description', 'Not provided')}
"""
        
        user_prompt = f"""Please create a comprehensive summary based on the following information:

{claim_summary}

Lab Test Results (Extracted Data):
{lab_test_data}

Lab Report Text (Full):
{lab_report_text[:5000]}...

Please provide a comprehensive summary that combines all this information, explains correlations, and provides medical insights."""
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }

        max_retries = 3
        base_timeout = 180
        
        response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)
                response = make_api_call(payload, headers, timeout=timeout)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                    else:
                        last_error = "Rate limited after all retries"
                        break
                else:
                    last_error = f"API error: {response.status_code}"
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    break
                    
            except Timeout:
                last_error = "Request timed out"
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            except RequestException as e:
                last_error = f"Network error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None

        if response and response.status_code == 200:
            try:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return content
            except (KeyError, IndexError, json.JSONDecodeError):
                return None
        else:
            return None

    except Exception as e:
        return None

def validate_question_relevance(question, claim_data=None, lab_report_text=None):
    """Validate if a question is related to lab report data, medical values, and their correlations"""
    
    def make_api_call(payload, headers, timeout=60):
        """Make API call with specified timeout"""
        return requests.post(
            f"{AZURE_ENDPOINT}/openai/deployments/PM-Business-2/chat/completions?api-version=2024-12-01-preview",
            headers=headers,
            json=payload,
            timeout=timeout
        )
    
    try:
        # Create context from available data
        context_parts = []
        
        if lab_report_text:
            context_parts.append(f"Lab report text available ({len(lab_report_text)} characters)")
        
        # Add claim data context if available
        claim_context = ""
        if claim_data:
            diagnosis = claim_data.get('diagnosis', '')
            procedure = claim_data.get('procedure_description', '')
            if diagnosis or procedure:
                claim_context = f"\nClaim Medical Information Available:\n- Diagnosis: {diagnosis or 'Not provided'}\n- Procedure: {procedure or 'Not provided'}\n- Primary ICD Code: {claim_data.get('primary_icd_code', 'Not provided')}\n- Secondary ICD Code: {claim_data.get('secondary_icd_code', 'Not applicable' if not claim_data.get('secondary_icd_code') else claim_data.get('secondary_icd_code'))}"
            context_parts.append("Claim medical information available")
        
        lab_summary = " | ".join(context_parts) if context_parts else "Lab report and claim information available"
        
        system_prompt = """You are a question validator for a medical lab report analysis system. 

Your task is to determine if a user's question is related to:
1. Lab report data, test results, and medical values
2. Medical values and their interpretation
3. Correlations between different lab values
4. What the lab test results mean medically
5. How lab results relate to the diagnosis, ICD codes, or procedures provided in the claim data
6. Questions connecting claim medical information (diagnosis, procedure, ICD codes) with lab report findings
7. Questions about the diagnosis, procedure, or medical condition itself (as long as related to the provided information)

Questions that are VALID:
- Questions about specific lab test values and what they mean
- Questions about correlations between different test results
- Questions about abnormal values and their implications
- Questions about reference ranges and normal vs abnormal
- Questions about what specific test results indicate
- Questions about how lab results relate to the diagnosis provided
- Questions about how lab results relate to the procedure performed
- Questions about ICD codes in relation to lab findings
- Questions connecting claim medical data (diagnosis, procedure, illness) with lab report results
- Questions asking why certain lab values might be abnormal given the diagnosis/procedure
- Questions about the diagnosis, procedure, or illness details provided

Questions that are INVALID (OUT OF CONTEXT):
- Questions about general medical treatments or medications not related to the provided data
- Questions about insurance claims processing or administrative matters
- Questions about unrelated medical topics not connected to the provided lab report or claim information
- Questions about diet, lifestyle, or general health advice unrelated to the provided information
- Questions that don't relate to the actual lab test data or claim information provided

IMPORTANT: If a question asks about the diagnosis, procedure, ICD codes, or illness details (even without explicit lab mention), it is VALID if that information was provided.
If a question asks about lab results in context of the diagnosis or procedure, it is VALID.

Respond ONLY with "VALID" or "INVALID" (in all caps at the start) followed by a brief one-sentence explanation. Do not provide any other response."""
        
        user_prompt = f"""Available Information: {lab_summary}{claim_context}

User Question: "{question}"

Is this question related to the lab report data, medical values, their correlations, or the claim medical information (diagnosis, procedure, ICD codes) provided?"""
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }

        try:
            response = make_api_call(payload, headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                content_upper = content.upper()
                
                # More robust validation check - look for VALID anywhere in first few words
                is_valid = (
                    content_upper.startswith('VALID') or
                    'VALID' in content_upper[:20] or  # Check first 20 chars
                    content_upper.startswith('YES') or
                    ('VALID QUESTION' in content_upper[:30])
                )
                
                # Also check if it's clearly INVALID
                if content_upper.startswith('INVALID') or 'INVALID' in content_upper[:20]:
                    is_valid = False
                
                return is_valid, content
            else:
                # Default to valid on API error to avoid blocking legitimate questions
                return True, "Validation API error - allowing question"
        except Exception:
            # Default to valid on exception to avoid blocking legitimate questions
            return True, "Validation error - allowing question"

    except Exception as e:
        # Default to valid on outer exception to avoid blocking legitimate questions
        return True, "Validation error - allowing question"

def answer_question_with_gpt(question, claim_data, uploaded_file_data=None):
    """Answer a validated question using GPT with claim data and any uploaded files"""
    
    def make_api_call(payload, headers, timeout=120):
        """Make API call with specified timeout"""
        try:
            return requests.post(
                f"{AZURE_ENDPOINT}/openai/deployments/PM-Business-2/chat/completions?api-version=2024-12-01-preview",
                headers=headers,
                json=payload,
                timeout=timeout
            )
        except Exception as e:
            raise RequestException(f"Failed to make API call: {str(e)}")
    
    # Validate inputs
    if not question or not question.strip():
        return "Please provide a valid question."
    
    # Ensure claim_data is a dict
    if not isinstance(claim_data, dict):
        claim_data = {}
    
    # Validate uploaded_file_data structure
    validated_file_data = None
    if uploaded_file_data:
        try:
            if isinstance(uploaded_file_data, tuple) and len(uploaded_file_data) >= 3:
                file_content = uploaded_file_data[0]
                file_type = uploaded_file_data[1]
                file_name = uploaded_file_data[2]
                mime_type = uploaded_file_data[3] if len(uploaded_file_data) >= 4 else None
                
                # For images, validate base64; for PDFs, validate text
                if file_type == "image":
                    if file_content and isinstance(file_content, str) and len(file_content) > 0:
                        validated_file_data = (file_content, file_type, file_name, mime_type or "image/jpeg")
                else:
                    # For PDFs and other files, ensure text exists
                    if file_content and str(file_content).strip():
                        validated_file_data = (str(file_content).strip(), file_type, file_name)
        except Exception:
            validated_file_data = None
    
    try:
        system_prompt = """You are a medical assistant answering questions via WhatsApp. Provide SHORT, CRISP, and CONCISE responses suitable for WhatsApp messaging.

CRITICAL FORMATTING REQUIREMENTS:
- Keep responses SHORT: 2-4 sentences maximum, ideally 50-100 words
- Write in a conversational, friendly WhatsApp style
- Use simple, plain language - avoid medical jargon
- Be direct and to-the-point
- Use line breaks for readability if needed
- Can use simple bullet points (•) if listing items
- NO long paragraphs or detailed explanations unless absolutely necessary

Guidelines:
- Base answers on the provided patient medical information AND any uploaded documents/attachments
- When a file/attachment is provided, you MUST analyze its content and incorporate it into your answer
- If a question is asked WITH an attachment, answer based on BOTH the question AND the attachment content
- For PDF attachments: Extract and use information from the PDF text content provided
- For image attachments: Acknowledge the image and answer based on what information is available
- Combine information from patient medical data with attachment content when answering
- If information is not in the provided data or attachment, say so briefly
- Focus on answering the specific question asked, using all available context (patient data + attachments)
- Answer questions about diagnosis, procedure, ICD codes, or medical conditions based on all available information
- Be accurate and medically sound but very concise
- Use simple, everyday language patients can understand

Remember: This is WhatsApp - users expect quick, brief, easy-to-read messages. Be helpful but keep it SHORT. When attachments are provided, they are part of the context - use them!"""
        
        # Prepare claim context with full patient information
        claim_context = ""
        if claim_data and isinstance(claim_data, dict):
            # Check if there's any meaningful data
            has_data = any(
                str(v).strip() and str(v).lower() not in ['not provided', 'none', 'n/a', '']
                for v in claim_data.values()
            )
            if has_data:
                claim_context = f"""
=== PATIENT MEDICAL INFORMATION ===

Patient Name: {claim_data.get('patient_name', 'Not provided')}
Hospital: {claim_data.get('hospital', 'Not provided')}
Insurer Risk ID: {claim_data.get('insurer_risk_id', 'Not provided')}
Diagnosis: {claim_data.get('diagnosis', 'Not provided')}
Illness Details: {claim_data.get('illness_details', 'Not provided')}
Primary ICD Code: {claim_data.get('primary_icd_code', 'Not provided')}
Secondary ICD Code: {claim_data.get('secondary_icd_code', 'Not provided' if not claim_data.get('secondary_icd_code') else claim_data.get('secondary_icd_code'))}
Procedure Description: {claim_data.get('procedure_description', 'Not provided')}
Risks/Complications: {claim_data.get('risks_complications', 'Not provided')}
Care Instructions: {claim_data.get('care_instructions', 'Not provided')}
Additional Advice: {claim_data.get('additional_advice', 'Not provided')}
Specialist/Department: {claim_data.get('specialist', 'Not provided')}

"""
        
        # Prepare uploaded file context - CRITICAL: Always include if attachment exists
        uploaded_file_context = ""
        image_base64 = None
        image_mime_type = None
        is_image_attachment = False
        
        if validated_file_data:
            try:
                file_content = validated_file_data[0]
                file_type = validated_file_data[1]
                file_name = validated_file_data[2]
                
                if file_type == "pdf":
                    # Include full PDF text, but truncate if extremely long (keep more context)
                    # Increase limit to 8000 chars to preserve more content
                    if len(file_content) > 8000:
                        # Keep first 4000 and last 4000 chars for better context
                        file_text_display = file_content[:4000] + "\n\n[... middle portion of document ...]\n\n" + file_content[-4000:]
                    else:
                        file_text_display = file_content
                    uploaded_file_context = f"""
=== UPLOADED DOCUMENT (PDF) ===
File Name: {file_name}

PDF Content:
{file_text_display}

"""
                elif file_type == "image":
                    # Store image data for vision API
                    is_image_attachment = True
                    image_base64 = file_content
                    image_mime_type = validated_file_data[3] if len(validated_file_data) >= 4 else "image/jpeg"
                    uploaded_file_context = f"""
=== UPLOADED IMAGE ===
File Name: {file_name}

Note: User has uploaded an image file. Please analyze the image content and answer their question based on what you see in the image.

"""
                else:
                    # For other file types
                    uploaded_file_context = f"""
=== UPLOADED FILE ===
File Name: {file_name}
File Type: {file_type}

File Content/Note:
{file_content if len(file_content) < 1000 else file_content[:1000] + "..."}

"""
            except Exception as e:
                # If there's an error parsing uploaded file data, still note that an attachment exists
                uploaded_file_context = f"""
=== UPLOADED ATTACHMENT ===
Note: User has attached a file. Please acknowledge this in your response if relevant to their question.

"""
        
        # Construct user prompt - ensuring BOTH question and attachment are included
        context_parts = []
        if claim_context:
            context_parts.append(claim_context)
        if uploaded_file_context:
            context_parts.append(uploaded_file_context)
        
        context_section = "\n".join(context_parts) if context_parts else ""
        
        # Build comprehensive prompt ensuring both question and attachment are included
        prompt_parts = []
        
        if context_section:
            prompt_parts.append(context_section)
        
        # Always include the user's question
        prompt_parts.append(f"=== USER QUESTION ===")
        prompt_parts.append("")
        prompt_parts.append(question)
        
        # Add instructions based on what data is available
        instruction_text = "IMPORTANT: Provide a SHORT, CRISP WhatsApp-style answer (2-4 sentences, 50-100 words maximum). Be direct, conversational, and easy to read."
        
        if uploaded_file_context:
            instruction_text += " The user has attached a file/document - please analyze the content of the attachment and answer their question based on BOTH the attachment content AND the patient medical information provided above."
        elif claim_context:
            instruction_text += " Reference specific information from the patient medical data provided above."
        else:
            instruction_text += " If you don't have specific information, acknowledge that briefly."
        
        instruction_text += " Keep it brief and to the point."
        
        prompt_parts.append("")
        prompt_parts.append(instruction_text)
        
        user_prompt_text = "\n".join(prompt_parts)
        
        # Build payload - handle images using vision API format
        if is_image_attachment and image_base64:
            # Use vision API format with content array
            user_content = [
                {"type": "text", "text": user_prompt_text}
            ]
            # Add image to content array
            image_url = f"data:{image_mime_type};base64,{image_base64}"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            }
        else:
            # Standard text-only payload
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text}
                ]
            }

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }

        # Retry mechanism
        max_retries = 3
        base_timeout = 120
        response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt) if attempt > 0 else base_timeout
                response = make_api_call(payload, headers, timeout=timeout)
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            return content
                        else:
                            last_error = "Invalid response format from API"
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        last_error = f"Error parsing response: {str(e)}"
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        break
                elif response.status_code == 429:
                    last_error = "Rate limited - too many requests"
                    if attempt < max_retries - 1:
                        wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                    break
                elif response.status_code == 401:
                    last_error = "Authentication failed - Invalid API key"
                    break
                elif response.status_code == 403:
                    try:
                        error_detail = response.json().get('error', {}).get('message', '')
                        last_error = f"⚠️ Access Denied: Azure OpenAI network restrictions are blocking access. Please check Azure Portal → Networking settings to allow your IP address."
                    except:
                        last_error = "⚠️ Access Denied: Azure OpenAI network restrictions are blocking access. Please check Azure Portal → Networking settings."
                    break
                elif response.status_code == 400:
                    try:
                        error_detail = response.json().get('error', {}).get('message', 'Bad request')
                        last_error = f"Bad request: {error_detail}"
                    except:
                        last_error = f"Bad request (status {response.status_code})"
                    break
                else:
                    try:
                        error_detail = response.text[:200]
                        last_error = f"API error (status {response.status_code}): {error_detail}"
                    except:
                        last_error = f"API error (status {response.status_code})"
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    break
                
            except Timeout:
                last_error = f"Request timed out after {timeout} seconds"
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            except RequestException as e:
                last_error = f"Network error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    break
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    break

        # If we get here, all retries failed
        if last_error:
            # Show user-friendly error message
            if "Access Denied" in last_error or "network restrictions" in last_error:
                return last_error  # Show the detailed 403 message with instructions
            else:
                return f"I'm having trouble processing your question right now. {last_error}"
        else:
            return "I'm having trouble processing your question right now. Please try again in a moment."

    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."

def display_analysis_results(analysis_data):
    """Display the analysis results in a tabular format"""
    if not analysis_data:
        return None

    df_data = []
    for item in analysis_data:
        df_data.append({
            "Test Name": item.get("test_name", "N/A"),
            "Raw Name": item.get("raw_name", "N/A"),
            "Observed Value": item.get("observed_value", "N/A"),
            "Unit": item.get("unit", "N/A"),
            "Reference Range": item.get("reference_range", "N/A"),
            "Level": item.get("level", "N/A").upper(),
            "Category": item.get("category", "Others"),
            "Page": item.get("page", "N/A"),
            "Notes": item.get("notes", "")
        })

    return pd.DataFrame(df_data)

def main():
    st.set_page_config(
        page_title="Medical Assistant Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better chat interface
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = CHAT_STATES['WELCOME']
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'claim_data' not in st.session_state:
        # Randomly select a patient for this session
        selected_patient = get_random_patient()
        st.session_state.claim_data = get_claim_data_from_patient(selected_patient)
        st.session_state.selected_patient_full = selected_patient  # Store full patient data
    if 'current_uploaded_file' not in st.session_state:
        st.session_state.current_uploaded_file = None
    if 'current_uploaded_file_data' not in st.session_state:
        st.session_state.current_uploaded_file_data = None
    if 'chat_attachments' not in st.session_state:
        st.session_state.chat_attachments = []  # Store attachments in chat context
    
    # Title
    st.title("🤖 Medical Assistant")
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Render chat history
        render_chat_history()
        
        current_step = get_current_step()
        
        # Welcome step
        if current_step == CHAT_STATES['WELCOME']:
            if len(st.session_state.messages) == 0:
                welcome_msg = """Hello! 👋 

I'm your Medical Assistant. I'm here to help answer any questions about your medical information, diagnosis, procedure, or any documents you'd like to share.

Feel free to ask me anything related to your medical claim or upload documents/photos if you have questions about them! 📄📷"""
                add_message('bot', welcome_msg)
                st.session_state.current_step = CHAT_STATES['QA']
                st.rerun()
        
        # Q&A step
        elif current_step == CHAT_STATES['QA']:
            st.markdown("---")
            st.markdown("### 💬 Ask Your Question")
            
            # Main question input - primary focus
            question = st.text_input(
                "Type your question:",
                key="question_input",
                placeholder="e.g., What does my diagnosis mean? What is my procedure about? Explain my ICD code..."
            )
            
            # Optional file attachment section (collapsible)
            with st.expander("📎 Attach Files (Optional)", expanded=False):
                tab1, tab2, tab3 = st.tabs(["📄 PDF", "📷 Photo", "📸 Camera"])
                
                with tab1:
                    uploaded_pdf = st.file_uploader("Upload PDF Document", type="pdf", key="pdf_uploader", help="Optional: Upload a PDF document to ask questions about it")
                    if uploaded_pdf is not None:
                        if uploaded_pdf != st.session_state.get('current_uploaded_file'):
                            with st.spinner("Processing PDF..."):
                                result = process_uploaded_file(uploaded_pdf)
                                if result and result[0]:
                                    file_text, file_type, file_name = result[0], result[1], result[2]
                                    uploaded_file_data = (file_text, file_type, file_name)
                                    st.session_state.current_uploaded_file = uploaded_pdf
                                    st.session_state.current_uploaded_file_data = uploaded_file_data
                                    add_message('user', f"📎 Uploaded PDF: {file_name}")
                                    add_message('bot', f"✅ PDF received: {file_name}. You can ask questions about it.")
                                    st.rerun()
                
                with tab2:
                    uploaded_photo = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'], key="photo_uploader", help="Optional: Upload a photo/image to ask questions about it")
                    if uploaded_photo is not None:
                        if uploaded_photo != st.session_state.get('current_uploaded_file'):
                            result = process_uploaded_file(uploaded_photo)
                            if result and result[0]:
                                # For images, result is (base64_image, "image", filename, mime_type)
                                if len(result) >= 4:
                                    base64_image, file_type, file_name, mime_type = result[0], result[1], result[2], result[3]
                                    uploaded_file_data = (base64_image, file_type, file_name, mime_type)
                                else:
                                    # Fallback for old format
                                    file_text, file_type, file_name = result[0], result[1], result[2]
                                    uploaded_file_data = (file_text, file_type, file_name, uploaded_photo.type if len(result) == 4 else "image/jpeg")
                                st.session_state.current_uploaded_file = uploaded_photo
                                st.session_state.current_uploaded_file_data = uploaded_file_data
                                add_message('user', f"📷 Uploaded photo: {file_name}")
                                add_message('bot', f"✅ Photo received: {file_name}. You can ask questions about it.")
                                st.rerun()
                
                with tab3:
                    # Only show camera input if user explicitly wants to use it
                    if 'show_camera' not in st.session_state:
                        st.session_state.show_camera = False
                    
                    if not st.session_state.show_camera:
                        if st.button("📸 Activate Camera", key="activate_camera_btn", use_container_width=True):
                            st.session_state.show_camera = True
                            st.rerun()
                    else:
                        camera_photo = st.camera_input("Take a photo", key="camera_input", help="Take a selfie or photo using your camera")
                        if camera_photo is not None:
                            if camera_photo != st.session_state.get('current_uploaded_file'):
                                result = process_uploaded_file(camera_photo)
                                if result and result[0]:
                                    # For images, result is (base64_image, "image", filename, mime_type)
                                    if len(result) >= 4:
                                        base64_image, file_type, file_name, mime_type = result[0], result[1], result[2], result[3]
                                        uploaded_file_data = (base64_image, file_type, "Camera Photo", mime_type)
                                    else:
                                        # Fallback for old format
                                        file_text, file_type = result[0], result[1]
                                        uploaded_file_data = (file_text, file_type, "Camera Photo", camera_photo.type if len(result) == 4 else "image/jpeg")
                                    st.session_state.current_uploaded_file = camera_photo
                                    st.session_state.current_uploaded_file_data = uploaded_file_data
                                    st.session_state.show_camera = False  # Hide camera after capture
                                    add_message('user', "📸 Took a photo")
                                    add_message('bot', "✅ Photo captured! You can ask questions about it.")
                                    st.rerun()
                        
                        # Option to hide camera
                        if st.button("❌ Close Camera", key="close_camera_btn"):
                            st.session_state.show_camera = False
                            st.rerun()
            
            # Show currently attached file if any (UI only)
            # Also show chat attachments that are stored in context
            has_ui_attachment = st.session_state.current_uploaded_file_data is not None
            has_chat_attachments = st.session_state.get('chat_attachments') and len(st.session_state.chat_attachments) > 0
            
            if has_ui_attachment:
                # Handle both old format (3 items) and new format (4 items for images)
                file_data = st.session_state.current_uploaded_file_data
                if len(file_data) >= 3:
                    file_name = file_data[2]
                    file_type = file_data[1]
                else:
                    file_name = "Unknown"
                    file_type = "unknown"
                    
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"📎 File attached: {file_name} ({file_type.upper()})")
                with col2:
                    if st.button("Clear", key="clear_file"):
                        # Clear UI attachment but keep in chat context if already stored
                        st.session_state.current_uploaded_file = None
                        st.session_state.current_uploaded_file_data = None
                        # Force file uploader to reset by clearing its key
                        st.session_state.pdf_uploader = None
                        st.session_state.photo_uploader = None
                        st.session_state.camera_input = None
                        # Hide camera
                        st.session_state.show_camera = False
                        add_message('bot', "File attachment cleared from input. Previous attachments are still available in chat context.")
                        st.rerun()
            
            # Show info about stored chat attachments
            if has_chat_attachments and not has_ui_attachment:
                attachment_count = len(st.session_state.chat_attachments)
                st.info(f"💾 {attachment_count} file(s) stored in chat context - available for future questions")
            
            st.markdown("---")
            
            # Ask question button
            if st.button("Ask Question", type="primary", use_container_width=True):
                if question.strip():
                    # Capture question value before clearing
                    captured_question = question.strip()
                    
                    # Clear text input immediately by deleting the key
                    if 'question_input' in st.session_state:
                        del st.session_state.question_input
                    
                    # Store attachment in chat context before clearing UI
                    uploaded_file_data = st.session_state.get('current_uploaded_file_data')
                    if uploaded_file_data:
                        # Validate and store attachment in chat context for future questions
                        if isinstance(uploaded_file_data, tuple) and len(uploaded_file_data) >= 3:
                            file_content = uploaded_file_data[0]
                            file_type = uploaded_file_data[1]
                            
                            # Validate attachment
                            is_valid = False
                            if file_type == "image":
                                if file_content and isinstance(file_content, str) and len(file_content) > 0:
                                    is_valid = True
                            else:
                                if file_content and str(file_content).strip():
                                    is_valid = True
                            
                            if is_valid:
                                # Store in chat attachments for future questions
                                if 'chat_attachments' not in st.session_state:
                                    st.session_state.chat_attachments = []
                                st.session_state.chat_attachments.append(uploaded_file_data)
                                # Also store file info in message
                                file_name = uploaded_file_data[2] if len(uploaded_file_data) >= 3 else "Attachment"
                                add_message('user', captured_question, datetime.now().strftime("%H:%M"), file_info=f"📎 {file_name}")
                            else:
                                add_message('user', captured_question, datetime.now().strftime("%H:%M"))
                        else:
                            add_message('user', captured_question, datetime.now().strftime("%H:%M"))
                    else:
                        add_message('user', captured_question, datetime.now().strftime("%H:%M"))
                    
                    # Get claim data
                    # Ensure we use the same patient data throughout the session
                    claim_data = st.session_state.get('claim_data', {})
                    if not isinstance(claim_data, dict) or not claim_data:
                        # If claim_data is missing or empty, reinitialize with a random patient
                        selected_patient = get_random_patient()
                        claim_data = get_claim_data_from_patient(selected_patient)
                        st.session_state.claim_data = claim_data
                        st.session_state.selected_patient_full = selected_patient
                    
                    # Get all attachments from chat context (including current one and previous ones)
                    all_attachments = st.session_state.get('chat_attachments', [])
                    # Use the most recent attachment (or current one if still available)
                    validated_uploaded_data = None
                    if all_attachments:
                        # Use the most recent attachment
                        validated_uploaded_data = all_attachments[-1]
                    
                    # Always send question to GPT with attachment if present
                    with st.spinner("Getting answer..."):
                        try:
                            answer = answer_question_with_gpt(
                                captured_question,
                                claim_data,
                                validated_uploaded_data  # Pass attachment from chat context
                            )
                        except Exception as e:
                            answer = f"I'm having trouble processing your question. Error: {str(e)}. Please try again."
                            st.error(f"Error details: {str(e)}")
                    
                    # Add bot answer
                    timestamp = datetime.now().strftime("%H:%M")
                    add_message('bot', answer, timestamp)
                    
                    # Clear attachment UI (but keep in chat context)
                    st.session_state.current_uploaded_file = None  # Clear UI attachment display
                    st.session_state.current_uploaded_file_data = None  # Clear UI attachment data
                    # Clear file uploader widgets (UI only - chat context preserved)
                    if 'pdf_uploader' in st.session_state:
                        del st.session_state.pdf_uploader
                    if 'photo_uploader' in st.session_state:
                        del st.session_state.photo_uploader
                    if 'camera_input' in st.session_state:
                        del st.session_state.camera_input
                    # Hide camera after question submission
                    st.session_state.show_camera = False
                    
                    st.rerun()
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main() 
