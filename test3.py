import os
from google import genai
from google.genai import types

def generate():
    # Initialize the GenAI client
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Read the job description from the file
    with open("job_description.txt", "r") as jd_file:
        job_description = jd_file.read()

    # Get all resume files from the 'resume' folder
    resume_folder = "./resume"
    resume_files = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder) if f.endswith(".pdf")]

    # Dictionary to store results
    results = {}

    # Loop through each resume and generate a score
    for idx, resume_file in enumerate(resume_files, start=1):
        # Upload the resume file
        uploaded_file = client.files.upload(file=resume_file)

        # Prepare the content for email extraction
        email_extraction_content = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                ],
            ),
        ]

        # Configure the content generation for email extraction
        email_extraction_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""You are an email extraction assistant. Your task is to extract all email addresses from the provided resume. Follow these steps:
                1. Analyze the content of the resume.
                2. Identify all valid email addresses present in the text.
                3. Return the email addresses as a comma-separated list.

                Input format:
                RESUME

                Output format:
                Emails = "email1@example.com, email2@example.com"
                """),
            ],
        )

        # Extract emails from the resume
        print(f"Extracting emails from resume: {resume_file}")
        emails = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=email_extraction_content,
            config=email_extraction_config,
        ):
            if chunk.text:  # Ensure chunk.text is not None
                emails += chunk.text
        emails = emails.strip().replace("Emails = ", "").strip('"')

        # Prepare the content for scoring
        scoring_content = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"JOB DESCRIPTION = \"{job_description}\""),
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                ],
            ),
        ]

        # Configure the content generation for scoring
        scoring_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""You are a resume screening assistant. Your task is to evaluate a candidate’s resume against a given job description and generate a score that reflects how well the candidate matches the requirements of the role. Follow these steps:
                1. Review the Job Description:
                    • Identify key responsibilities, required skills, experiences, and education.
                    • Note any preferred qualifications or additional criteria.
                2. Analyze the Resume:
                    • Look for evidence of relevant experience, skills, and achievements that align with the job description.
                    • Identify any gaps or mismatches compared to the requirements.
                3. Generate a Score:
                    • Assign a numerical score between 0 and 100 based on how well the candidate meets the job requirements.
                4. Give the score as a final output.

                Input format: 
                JOB DESCRIPTION =\" \"
                RESUME

                Output format:
                Score = 
                """),
            ],
        )

        # Generate the score for the resume
# Generate the score for the resume
        print(f"Processing resume: {resume_file}")
        score = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=scoring_content,
            config=scoring_config,
        ):
            if chunk.text:  # Ensure chunk.text is not None
                score += chunk.text
        score = score.strip().replace("Score = ", "").strip()

        # Store the result in the dictionary
        results[f"id{idx}"] = [os.path.basename(resume_file), int(score), emails]

    # Sort the results dictionary by scores in descending order
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1][1], reverse=True))

    screened_resumes = sorted_results
    os.makedirs("data", exist_ok=True)  # Ensure the 'data' directory exists
    with open("data/common_data.txt", "w") as file:
        file.write(f"screened_resumes = {screened_resumes}")
    # Print the sorted results in the desired format
    print("\nSorted Results:")
    print(sorted_results)

if __name__ == "__main__":
    generate()