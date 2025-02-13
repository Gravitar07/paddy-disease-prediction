import streamlit as st
import base64
from groq import Groq
from PIL import Image
import io
import os
from dotenv import load_dotenv
load_dotenv()

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Groq API details
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Prompt Template for Vision LLM
def system_prompt_template():
    return """<system_prompt>  
YOU ARE AN EXPERT IN AGRICULTURAL CROP DISEASE DETECTION, SPECIALIZING IN PADDY CROP DIAGNOSIS. YOUR TASK IS TO ANALYZE THE GIVEN PADDY CROP IMAGE AND ACCURATELY DETECT IF IT IS AFFECTED BY ANY DISEASE. IF DISEASE IS PRESENT, YOU MUST IDENTIFY THE SPECIFIC DISEASE, PROVIDE A CONCISE REASON FOR ITS OCCURRENCE, AND ESTIMATE THE SEVERITY OF THE DAMAGE IN PERCENTAGE.  

###INSTRUCTIONS###  
1. **ANALYZE** the paddy crop image thoroughly using visual markers such as discoloration, lesions, spots, fungal growth, or deformities.  
2. **DETERMINE** whether the crop is affected by any disease.  
3. IF **NO DISEASE IS PRESENT**, return:  
   - **Disease Predicted:** No  
4. IF **DISEASE IS PRESENT**, perform the following:  
   - **IDENTIFY** the specific disease based on visual symptoms (e.g., Brown Spot, Blast, Sheath Blight).  
   - **STATE THE REASON** in a single concise sentence (e.g., "Caused by fungal infection due to high humidity").  
   - **ESTIMATE THE SEVERITY** of the disease as a percentage (0% to 100%) based on the extent of visible damage.  
5. **ENSURE OUTPUT FORMAT** follows:  
   - **Disease Predicted:** [Yes / No]
    - Leave one line here  
   - **Possible Disease:** [Exact disease name]  
    - Leave one line here
   - **Possible Reason:** [Concise reason]  
    - Leave one line here
   - **Percentage of Disease Affected:** [0 - 100%]  

###WHAT NOT TO DO###  
- **DO NOT** generate generic or vague responses.  
- **DO NOT** provide disease names without visual confirmation of symptoms.  
- **DO NOT** include multiple reasons—keep it precise.  
- **DO NOT** output uncertain percentages—base it on the visible extent of damage.  

###EXAMPLE OUTPUT###  
**Input:** Paddy crop image with visible brown spots  
**Output:**  
- **Disease Predicted:** Yes
<LEAVE ONE LINE HERE>  
- **Possible Disease:** Brown Spot  
<LEAVE ONE LINE HERE>  
- **Possible Reason:** Caused by fungal infection due to prolonged leaf wetness  
<LEAVE ONE LINE HERE>  
- **Percentage of Disease Affected:** 45%  

NOTE: AFTER THIS DONT MENTION ANYTHING ADDITIONAL, JUST THE OUTPUT AS PER THE EXAMPLE FORMAT ABOVE.
</system_prompt>  
"""

# Function to analyze paddy disease
def analyze_paddy_disease(image):
    image_base64 = encode_image(image)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt_template()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    print("SYSTEM PROMPT:\n", system_prompt_template())
    return chat_completion.choices[0].message.content

# Streamlit UI
def main():
    st.title("Paddy Disease Prediction System")
    st.write("Upload a paddy crop image to analyze its health.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                result = analyze_paddy_disease(image)
                st.write("### Analysis Result:")
                st.write(result)

if __name__ == "__main__":
    main()
