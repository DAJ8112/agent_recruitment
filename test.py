import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    files = [
        # Please ensure that the file is available in local system working direrctory or change the file path.
        client.files.upload(file="./4.pdf"),
    ]
    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""JOB DESCRIPTION = 
\"Position Summary

Sony Corporation of America (SCA) is seeking a Data Science Intern, Risk Analytics to join Sony’s Corporate Information Security Division. This position will report to the Director of Risk Analytics and be a part of the team responsible for designing, implementing, and maintaining a Global Information Security Risk Analytics platform used to provide information security risk intelligence to Senior Leadership and monitor the confidentiality, integrity, and availability of Sony assets.

The ideal candidate will be passionate about using data analytics and data science techniques to solve complex business problems. This role will involve hands-on experience in machine learning, statistical analysis, and data-driven decision-making to help shape our data strategy and drive insights from large, diverse datasets.

Job Responsibilities

Assist in the development and implementation of models to solve business challenges and enhance decision-making processes

Work with large datasets to perform data exploration, preprocessing, and feature engineering 

Contribute to data visualization and storytelling to present findings and recommendations to stakeholders

Analyze large amounts of data/information to discover and articulate trends and patterns

Required Skills/Experience

Working towards a graduate or undergraduate degree in Computer Science, Applied Mathematics, Statistics, Data Science/Analytics, or another related field

Available at least 20 hours per week 

Strong interest in data science and advanced analytics

Familiarity with data modelling, statistical methods, machine learning, algorithm development, and data mining techniques

Proficiency in Python, SQL, or R

Experience in data wrangling, data visualization, and exploratory data analysis (EDA)

Excellent problem-solving, analytical, and critical thinking skills\""""),
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are a resume screening assistant. Your task is to evaluate a candidate’s resume against a given job description and generate a score that reflects how well the candidate matches the requirements of the role. Follow these steps:
	1.	Review the Job Description:
	    •	Identify key responsibilities, required skills, experiences, and education.
	    •	Note any preferred qualifications or additional criteria.
	2.	Analyze the Resume:
	    •	Look for evidence of relevant experience, skills, and achievements that align with the job description.
	    •	Identify any gaps or mismatches compared to the requirements.
	3.	Generate a Score:
	    •	Assign a numerical score between 0 and 100 (with 100 indicating an ideal match) based on how well the candidate meets the job requirements.
    4. give the score as a final output.

    Input format: 
    JOB DESCRIPTION =\" \"
    RESUME

    Ouput format:
    Score = \"\"
    """),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
