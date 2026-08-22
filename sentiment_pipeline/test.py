# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client(
#     api_key = os.getenv("GEMINI_API_KEY"),
# )


# generation_configs = {
#     'temperature' : 1,
#     'max_output_token' : 65536,
#     'top_p' : 0.95,
#     'thinking_level' : 'low'
# }

# interation = client.interactions.create(
#     model = 'models/gemini-3.7-flash',
#     input = 'How does LLMs Work?',
#     generation_config=generation_configs
# )

# print(interation.steps[-1].content[0].text)


for i in range(1,7):
    print(i)