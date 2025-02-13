# AI-Based Paddy (Blast Disease) Detection System

This project leverages the power of Generative AI to analyze paddy crop images for disease detection. The model identifies Blast Disease, which is a significant threat to rice crops. Using a deep learning-based Vision LLM, the system can classify the health of the crop, provide the potential disease name, its cause, and estimate the severity of the damage.

## Key Features
- **Disease Detection:** Automatically detects Blast Disease or other diseases in paddy crops.
- **Image Upload and Analysis:** Allows users to upload paddy crop images and receive detailed analysis results.
- **Severity Estimation:** Provides a severity percentage to quantify the extent of disease damage.
  
## Procedure
### Prerequisites
Before running the project, ensure that you have the following:
- **Python 3.7+** installed (Python 3.10 is recommended)
- **Groq API Key** (for accessing the AI model)

### Installation Steps

1. **Clone the repository (if applicable)** or simply download the project files to your local machine.

2. **Install the Required Dependencies**:
   Navigate to the project directory where the `requirements.txt` is located and run:
   ```bash
   pip install -r requirements.txt

3. **Run the Application**: You can now run the app by executing the following command:
    ```bash
    streamlit run app.py

4. **Open the App**: Streamlit will open a browser window with the web app interface. You can upload paddy crop images and analyze them.

## Modules Used
1. **Streamlit**: A popular Python framework to build interactive web applications for machine learning projects.
    - Version: 1.14.0 or later

2. **Python-dotenv**: A Python package that reads key-value pairs from a .env file and sets them as environment variables.
    - Version: 0.19.2

3. **Groq**: The core library to interact with the Groq API, providing access to advanced models for vision-based AI tasks.
    - Version: 0.1.0

4. **Pillow (PIL)**: Python Imaging Library, used to open, manipulate, and save image files.
    - Version: 8.4.0 or later

## Model Used
The AI model used in this project is based on Llama Vision LLM, specifically the llama-3.2-11b-vision-preview through Groq API Platform. This model is tailored for analyzing visual data, such as images of paddy crops, and performing detailed analysis to detect diseases like Blast Disease.

## Capabilities of the Model:
1. **Disease Identification**: The model detects whether the paddy crop is affected by Blast Disease or another type of disease.

2. **Cause Analysis**: It provides insights into the potential causes of the detected disease, such as environmental factors (e.g., humidity, soil conditions).

3. **Severity Estimation**: Based on visible damage, the model estimates the percentage of crop affected by the disease.

4. **Precision**: The model is designed to provide highly accurate results by closely examining visual markers like lesions, discoloration, fungal growth, and other symptoms of Blast Disease.

### Example Use Case
- Upload Image: The user uploads an image of a paddy crop showing visible symptoms of disease.
- Analyze Image: Once uploaded, the system processes the image using the Groq model and generates a detailed report.

### Output: The output includes:
- Whether the disease is detected
- The type of disease (e.g., Blast Disease)
- The cause of the disease
- The severity of the damage (percentage)

### Example Output:
- **Disease Predicted:** Yes

- **Possible Disease:** Blast Disease

- **Possible Reason:** Caused by the Magnaporthe oryzae fungus due to high humidity and nitrogen-rich soil

- **Percentage of Disease Affected:** 60%