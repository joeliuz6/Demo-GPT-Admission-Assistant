import gradio as gr
import time
from openai import OpenAI
import pdfplumber
import os 



client = OpenAI(api_key="")  # your openai api key

summary_instruction = '''Using the provided college transcript, summarize the student's performance in mathematics. 

Include the following details:

0.Country, Institution, major (latest ranking, type of institution)

1.List all relevant Mathematics courses along with the grades received (e.g., Calculus I: A, Algebra II: B+).

2.Perform a statistical analysis of the math course grades. Calculate the average grade, the standard deviation, the range of grades, and the total number of math courses taken.

3.Indicate if any Math courses were taken during the COVID-19 pandemic periods (Fall 2020, Spring 2021, Fall 2021).

5.Note any transfer or AP Math courses, specifying if they were from USA or international institutions.

6.Ensure anonymization of the student's name and any personal identifiers.

Finally, provide a concise evaluation of the student's overall performance in their Math courses, based on the above data. 
Rate the student's performance as Excellent, Good, Average, or Poor (1-10).'''


def pdf_to_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text from the page
            page_text = page.extract_text()
            if page_text:  # Ensure there's text on the page
                text += page_text + "\n"
    return text

#gpt-4-0613
#gpt-3.5-turbo-1106
def get_summary_chat4(intro,model="gpt-3.5-turbo-1106"):

    """Given the transcript of the student, return the summary of the student performance in the college."""
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant to rate the performance of the university graduate program applicant."},
        {"role": "user", "content": summary_instruction + intro}
    ],
    temperature=0.7,
    )
    
    return response.choices[0].message.content

class PDFChatbot:
    def __init__(self):
        self.pdf_text = ""  # Variable to store PDF text

    def process_pdf(self, file_content):

        if not os.path.exists("cache"):
            os.makedirs("cache")
        file_name = f"cache/{time.time()}.pdf"
        with open(file_name, "wb") as f:
            f.write(file_content)



        assert file_name.endswith(".pdf")

        text_data = pdf_to_text(file_name)

        self.pdf_text = text_data
        

        print(text_data)

        try: 
            print(f"Getting summary...")
            summary = get_summary_chat4(text_data)
        except Exception as e:
            return f"Failed to get summary... Error: {e}"

        return summary

    def chat(self, user_input):
        context = f"The following is a summary of the student's academic performance based on their transcript: {self.pdf_text}"
        combined_input = context + "\n\n" + user_input
        print(combined_input)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant to answer questions given a student transcript."},
                    {"role": "user", "content": combined_input}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

def main():
    chatbot = PDFChatbot()

    # Custom CSS for vertical layout and background
    custom_css = """
    .gradio-container {
        background: url('https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/RushRheesLibraryInWinterObliqueFromLeft.jpg/2560px-RushRheesLibraryInWinterObliqueFromLeft.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    .gradio-app > div {
        flex-direction: column;
    }
    """

    # Interface for PDF processing
    pdf_interface = gr.Interface(
        fn=chatbot.process_pdf,
        inputs=gr.File(label="Upload Transcript", type="binary"),
        outputs=gr.Textbox(label="Student Transcript Summary"),
        css=custom_css
    )

    # Interface for the chatbot
    chat_interface = gr.Interface(
        fn=chatbot.chat,
        inputs=gr.Textbox(label="Ask a question about the student"),
        outputs=gr.Textbox(label="Answers according to the student transcript"),
        css=custom_css
    )

    # Tabbed interface combining both
    demo = gr.TabbedInterface(
        [pdf_interface, chat_interface], 
        ["Transcript Summarization", "Chatbot"],
        title="College Admission Assistant",
        css=custom_css
    )

    demo.launch(server_name="127.0.0.1", server_port=7799)

if __name__ == "__main__":
    main()
