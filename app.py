import gradio as gr
import re

from model import VideoProcessor
from pdf_utils import generate_full_pdf, generate_user_notes_pdf


url_temp = ""
docs = None
total_info = {
    "summary": None,
    "notes": None,
    "questions": None,
    "answers": None,
    "Chat History": [],
    "Mind Map": None,
}

processor = VideoProcessor()


def dict_to_pdf():
    global total_info
    fpath = generate_full_pdf(total_info)
    return fpath


def get_transcript(url):
    global docs
    if url is None:
        global url_temp
        url = url_temp
    video_id = re.findall(r"v=([^&]+)", url)[0]
    summary, docs = processor.get_summary(video_id)
    total_info["summary"] = summary
    return summary


def get_mind_map(url=None):
    if url is None:
        global url_temp
        url = url_temp
    global docs
    video_id = re.findall(r"v=([^&]+)", url_temp)[0]
    mindmap, _ = processor.get_mindmap(video_id, docs)
    total_info["Mind Map"] = mindmap
    return mindmap


def get_notes(url=None):
    if url is None:
        global url_temp
        url = url_temp
    global docs
    video_id = re.findall(r"v=([^&]+)", url_temp)[0]
    notes, _ = processor.get_notes(video_id, docs)
    global total_info
    total_info["notes"] = notes
    return notes


def embed_youtube(url):
    global url_temp
    url_temp = url
    video_id = re.findall(r"v=([^&]+)", url)[0]
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    embed_html = f'<div style="display: flex; justify-content: center;"><iframe width="600" height="315" src="{embed_url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>'
    return embed_html


def get_chats(question, url):
    video_id = re.findall(r"v=([^&]+)", url)[0]

    global docs
    answer, _ = processor.get_answer(video_id, question, docs)

    return answer


global answers


def get_questions_callback(url=None):
    if url is None:
        global url_temp
        url = url_temp

    video_id = re.findall(r"v=([^&]+)", url_temp)[0]
    result, _ = processor.get_question_answers(video_id, docs)
    questions = result["questions"]
    global answers
    answers = result["answers"]

    global total_info
    total_info["answers"] = answers
    total_info["questions"] = questions

    return questions


def get_answers_callback():
    global answers
    return answers


with gr.Blocks(
    css="footer {visibility: hidden}", theme="Base", title="Study Aid"
) as demo:
    gr.Markdown(" # <center> Study Aid")
    with gr.Tab("Learning"):
        text_input = gr.Textbox(
            label="Paste the YouTube URL for the video you're studying or have questions about"
        )

        with gr.Row():
            with gr.Column():
                html_output = gr.outputs.HTML()
                text_button = gr.Button("Enter your URL")
            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label="Enter question for chatbot")
                clear = gr.ClearButton([msg, chatbot])

        text_input.submit(embed_youtube, inputs=text_input, outputs=html_output)
        text_button.click(embed_youtube, inputs=text_input, outputs=html_output)

        trs_output = gr.outputs.Textbox(label="Summary")
        b2 = gr.Button("Get lecture summary")
        b2.click(get_transcript, inputs=text_input, outputs=trs_output)

        def respond(question, chat_history):
            bot_message = get_chats(question, url_temp)
            chat_history.append((question, bot_message))

            global total_info
            total_info["Chat History"].append(question)
            total_info["Chat History"].append(bot_message)

            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    with gr.Tab("Mind Map"):
        mm_output = gr.Plot(label="Mind Map")
        b2 = gr.Button("Get Memory Map")
        b2.click(get_mind_map, inputs=[], outputs=mm_output)

    with gr.Tab("Lecture Notes"):
        notes_output = gr.Markdown(label="Notes")
        notes_button = gr.Button("Generate Notes!")
        notes_button.click(get_notes, inputs=[], outputs=notes_output)

    with gr.Tab("Test Your Knowledge"):
        questions_button = gr.Button("Start testing my knowledge!!")
        question_boxes = []
        answer_boxes = []
        for _ in range(5):
            with gr.Row():
                question_boxes.append(gr.Textbox(label="Question"))
                answer_boxes.append(gr.Textbox(label="Answer"))
        quiz_button = gr.Button("Get Answers")
        questions_button.click(fn=get_questions_callback, outputs=question_boxes)
        quiz_button.click(fn=get_answers_callback, outputs=answer_boxes)

    with gr.Tab("Downloads"):
        pdf_output = gr.outputs.File(label="Download PDF")
        b2 = gr.Button("Get PDF")
        b2.click(dict_to_pdf, inputs=[], outputs=pdf_output)

    with gr.Tab("Write Notes"):
        pdf_inp = gr.inputs.Textbox(label="Enter your text here", lines=10)
        pdf_notes_output = gr.outputs.File(label="Download PDF")
        b2_notes = gr.Button("Get PDF")
        b2_notes.click(
            generate_user_notes_pdf, inputs=pdf_inp, outputs=pdf_notes_output
        )


demo.launch()
