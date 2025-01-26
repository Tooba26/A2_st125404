import streamlit as st

fa_css = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
<i class="fab fa-github"></i>
''' 

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-30px);
        }
        60% {
            transform: translateY(-15px);
        }
    }

     /* Typewriter animation */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blink {
        from, to { border-color: transparent; }
        50% { border-color: black; }
    }

    .stApp {
        background: -webkit-linear-gradient(to right, #24243e, #302b63, #0f0c29);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #24243e, #302b63, #0f0c29); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
        color: white; /* Optional: Change text color to match the background */
    }

    .typewriter {
        font-size: 24px;
        font-weight: bold;
        font-family: monospace;
        white-space: nowrap;
        overflow: hidden;
        border-right: 2px solid black;
        width: 100%; /* Adjust to control the container width */
        animation: typing 4s steps(30, end), blink 0.5s step-end infinite;
        word-wrap: break-word;
    }

    [data-testid="stSidebar"] {
        background-color: #1f1f3f; /* Matching sidebar color */
        color: #ffffff; /* White text for contrast */
    }
    

    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Navigation")  # This will always appear first
    st.markdown("Choose an option from above option ⬆️")

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default page is "home"

# Function to change the page
def navigate_to(page):
    st.session_state.page = page



# Home Page content
if st.session_state.page == "home":
    st.markdown("# 🏠 Home Page")
    st.write('<div class="typewriter">Welcome To the Language Model Web App</div>', unsafe_allow_html=True)
    st.write("The Web App have two models, a Book Corpus and Switchboard-1 Release 2")
    st.write("Select any model from the side bar..")


import streamlit as st





