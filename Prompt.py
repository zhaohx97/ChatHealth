import pandas as pd 
import os
import base64
from tqdm import tqdm

def run_reasoning_prompt_GPT(df_input, client, prompt_type, model_name="gpt-4o", save_path="Result"):

    answers_list = []
    df_output = None

    print(f"\nRunning for prompt_type: {prompt_type}...")

    for i, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Prompt: {prompt_type}"):
        img_path = row['img']

        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
        file_ext = os.path.splitext(img_path)[1][1:]
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        image_url = f"data:image/{file_ext};base64,{img_base64}"

        percent_65 = row['percent_65_years_and_over']
        percent_male = row['percent_male']
        percent_black = row['percent_black_or_african_american']
        percent_poverty = row['percent_poverty']
        percent_bachelor = row['percent_bachelor']

        socioeconomic_info = (
            f"- Poverty rate: {percent_poverty}%\n"
            f"- Share of residents with a bachelor's degree or higher: {percent_bachelor}%\n"
        )
        demographic_info = (
            f"- Share of residents aged 65 and older: {percent_65}%\n"
            f"- Male population percentage: {percent_male}%\n"
            f"- Black or African American population percentage: {percent_black}%\n"
        )

        if prompt_type == 'img_only':
            sociodemographic_info = ""
        elif prompt_type == 'socioeconomic':
            sociodemographic_info = "Socioeconomic characteristics of the neighborhood:\n" + socioeconomic_info + "\n"
        elif prompt_type == 'demographic':
            sociodemographic_info = "Demographic characteristics of the neighborhood:\n" + demographic_info + "\n"
        elif prompt_type == 'all':
            sociodemographic_info = "Socioeconomic and demographic characteristics of the neighborhood:\n" + socioeconomic_info + demographic_info + "\n"
        else:
            raise ValueError("Invalid prompt_type.")

        system_prompt = (
            "You are an expert in computer vision, public health, and urban informatics,"
            "specializing in evaluating how built environments inferred from satellite imagery"
            "influence neighborhood-level health outcomes."
        )

        if prompt_type == 'img_only':
            user_prompt = (
                "Carefully examine the satellite image below and estimate the likelihood"
                "that residents in this area have insufficient leisure-time physical activity."
                "Provide a score on a 1-10 scale:\n\n"
            )
        else:
            user_prompt = (
                "Carefully examine the satellite image below and, considering the provided neighborhood attributes,"
                "estimate the likelihood that residents in this area have insufficient leisure-time physical activity."
                "Provide a score on a 1-10 scale:\n\n"
            )

        user_prompt += (
            "1 = very low likelihood of physical inactivity (residents are highly active)\n"
            "10 = very high likelihood of physical inactivity (residents are largely inactive)\n\n"
            f"{sociodemographic_info}"
            "Please respond strictly in the following structure:\n"
            "[Score]:\n"
            "An integer between 1 and 10\n"
            "[Justification]:\n"
            "A brief explanation in English (1-3 sentences) describing your reasoning.\n"
            "Please strictly follow this format to facilitate automated data extraction."
        )

        response = client.chat.completions.create(
            model = model_name,
            temperature = 0.1,
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
        )

        answer = response.choices[0].message.content
        answers_list.append({
            "index": i,
            f"answer_{prompt_type}": answer
        })

    df_answers = pd.DataFrame(answers_list)
    df_output = df_input.reset_index().merge(df_answers, left_on='index', right_on='index', how='left')

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"LLMOut_NYC_{prompt_type}_GPT.csv")
    df_output.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    return answers_list, df_output
