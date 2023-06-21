# LLMs
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Streamlit
import streamlit as st

# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')

##=================================
##     S T R E A M L I T
##=================================
st.header("🚀 3-2-1 AI Newsletter")
st.markdown("Inspired by James Clear's 3-2-1 newsletter, I thought it would be fun to create an app that writes a weekly newsletter for me. \
                This tool scrapes from recent research papers published and Youtube videos to provide an update of the past week.\
                \n\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#) [markdownify](https://pypi.org/project/markdownify/) [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@TrentSwords](https://www.linkedin.com/in/trentswords/). \n Big thanks to [@GregKamradt](https://twitter.com/GregKamradt) for his online tutorials!")

st.image(image='AI Bot Writing Newsletter2.png', width=300, caption='Mid Journey: AI bot writing newsletter in style of cartoon')

st.subheader("🤖 How the newsletter is made?")
st.markdown(
"""
- **3 Papers** - scraped ML papers trending on Github
- **2 Quotes** - from interesting Youtube videos I bookmarked in Notion
- **1 Question** - created by yours truly - ChatGPT ;)
""")
st.image(image='Figure.png', width=500)

st.subheader("✏️ Let's get writing...")
st.markdown("Click the button below to tell Mr. Robot to get writing.")
st.markdown("\n\n")
button_ind = st.button("*Generate this week's newsletter*", type = 'secondary', help = "Click to generate output based on information.")

if button_ind:
    ##=================================
    ##     P A P E R S
    ##=================================
    def extract_ml_papers(html):
        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Find the table containing the ML papers
        table = soup.find('table')

        # Initialize a string to store the extracted data
        paper = "RESEARCH PAPERS.  "

        # Check if the table exists
        if table is None:
            print("No table found.")
            return result

        # Iterate over each row in the table
        for row in table.find_all('tr')[1:]:
            # Extract the paper name, description, and link
            columns = row.find_all('td')

            # Check if the columns exist
            if len(columns) < 2:
                print("Invalid table structure. Skipping row.")
                continue

            paper_name_element = columns[0].strong
            paper_link_element = columns[1].find('a')

            # Check if the necessary elements exist
            if paper_name_element is None or paper_link_element is None:
                print("Missing paper name or link. Skipping row.")
                continue

            paper_name = paper_name_element.get_text(strip=True)
            paper_description = columns[0].get_text(strip=True)[len(paper_name) + 3:].strip(') ')
            paper_link = paper_link_element['href']

            # Remove newline characters from the extracted data
            paper_name = paper_name.replace('\n', '')
            paper_description = paper_description.replace('\n', '')
            paper_link = paper_link.replace('\n', '')

            # Add the extracted data to the result string
            paper += f"Name: {paper_name} "
            paper += f"Description: {paper_description} "
            paper += f"Link: {paper_link} "
        #         paper += "---\n"

        return paper


    st.write("Scraping from papers...")
    url = "https://github.com/dair-ai/ML-Papers-of-the-Week/blob/main/README.md#"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    html = str(soup)
    paper = extract_ml_papers(html)
    paper = md(paper)

    ##=================================
    ##    Y O U T U B E
    ##=================================
    # Pulling data from YouTube in text form
    def get_video_transcripts(url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        documents = loader.load()
        transcript = ' '.join([doc.page_content for doc in documents])
        return transcript


    st.write("Scraping from Youtube...")
    video_urls = ['https://www.youtube.com/watch?v=6mdLXUbcw2Y&ab_channel=GregKamradt%28DataIndy%29']
    videos_text = ""

    for video_url in video_urls:
        video_text = get_video_transcripts(video_url)
        videos_text += video_text

    video_text = "YOUTUBE TRANSCRIPT.  " + video_text.replace('\n', '').replace('\xa0', '').strip()
    video_text = video_text[:20000]

    ##=================================
    ##   COMBINING + SPLITTING
    ##=================================
    user_information = paper + video_text #  + user_tweets

    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([user_information])


    ##=================================
    ##   PROMPTING
    ##=================================
    map_prompt = """You are a helpful AI bot that aids in condensing news information.
    Below is information about recent research, developments, and videos in the AI space to help create a newsletter.
    
    % START OF INFORMATION:
    {text}
    % END OF INFORMATION:
    
    Using the list of research papers from the past week, select the 3 most interesting papers from the list and rank in order of importance.
    For each paper, provide a 1 sentence description of why the findings in the paper are important.
    It is VERY important to keep the link for each research paper.
    
    From the Youtube video transcript, find two thought-provoking quotes. The quote must come from the Youtube video.
    For each quote, list the quote and identify who said it.
    
    Create one insightful question to pose to the newsletter audience that is relevant to information above.
    
    YOUR RESPONSE:"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])


    combine_prompt = """
    You are FormatGPT, an AI bot that helps me get text into a desired format.
    You will be given information from research papers, quotes from a Youtube video, and an insightful question.
    
    Here is an example of my intended format to be used in a Streamlit app:
    
    **3 PIONEERING PAPERS**
    I. FinGPT: Open-Source Financial Large Language Models. 
        FinGPT provides an open-source alternative to proprietary financial language models, allowing for democratization of Internet-scale financial data.
        [Link](https://arxiv.org/pdf/2306.06031v1.pdf)
    II. Algorithmic Trading Language Modelling
        This paper explores the application of language modeling techniques to algorithmic trading, offering potential advancements in financial prediction and decision-making.
        [Link](https://arxiv.org/pdf/2301.11325.pdf)
    III. WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences
        WebGLM presents a web-enhanced question-answering system based on the General Language Model (GLM), aiming to improve the efficiency and accuracy of answering user queries on the web.
        [Link](https://arxiv.org/pdf/2306.07906.pdf)
    
    **2 QUOTES FROM OTHERS**
    I. "In 2022, half of A.I. researchers stated in a survey that they believed there is a 10% or greater chance that humans go extinct from our inability to control AI.  If the people who are developing A.I. believe this, why aren’t we listening?" ~ Tristan Harris
    II. "While we've been fixated on the point technology overwhelms human strength, we've missed the point when technology undermines human weakness. That's the true 'singularity'." ~ Aza Raskin
    
    **1 QUESTION FOR YOU**
    How can we ensure that AI technologies are developed and deployed in a way that aligns with our human values and priorities?"
    %END:
    
    Using the text below, convert it to be in the same format as the text above.
    It is VERY important to include the link to each research paper:
    
    {text}
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    llm = ChatOpenAI(temperature=0.05, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
    # 'text-davinci-003'

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
    #                              verboseharri=True
                                )

    st.write("Sending to LLM...")
    output = chain({"input_documents": docs# The seven docs that were created before
    #                 ,"persons_name": "Elad Gil"
                   })

    st.markdown(f'#### Output:')
    st.write(output['output_text'])