# Import necessary packages
from ibm_watson_machine_learning.foundation_models import Model
import gradio as gr


model_id = "meta-llama/llama-2-13b-chat"  # Directly specifying the LLAMA2 model

# Set credentials to use the model
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

# Generation parameters
gen_parms = {
    "max_new_tokens": 512,  # Increased token limit for larger content
    "temperature": 0.7  # Adjusted for more creative variations
}
project_id = "skills-network"  # Specifying project_id as provided
space_id = None
verify = False

# Initialize the model
model = Model(model_id, my_credentials, gen_parms, project_id, space_id, verify)

# Function to polish the resume using the model, making polish_prompt optional
def customize_cover_letter(company_name, position_name, job_description, resume_content):
    # create prompt accordingly
    prompt_use = f"Given the resume content: '{resume_content}', create customized cover letter based on  job with job description {job_description} for the {position_name} position at {company_name} organization."
    
    
    # Generate a response using the model with parameters
    generated_response = model.generate(prompt_use)
    
    # Extract and return the generated text
    generated_text = generated_response["results"][0]["generated_text"]
    return generated_text

# Create Gradio interface for the customize cover letter
customize_cover_letter_application = gr.Interface(
    fn=customize_cover_letter,
    allow_flagging="never", # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(label="Company Name", placeholder="Enter the name of the company..."),
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        gr.Textbox(label="Job Description", placeholder="Enter the job description..."),
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=20),
    ],
    outputs=gr.Textbox(label="Cover letter"),
    title="Customized Cover Letter",
    description="This application helps you create customized cover letter for you. Enter the company , position name, job description and resume content"
)

# Launch the application
customize_cover_letter_application.launch()