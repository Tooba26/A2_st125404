import streamlit as st

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default page is "home"

# Function to change the page
def navigate_to(page):
    st.session_state.page = page

# Home Page content
if st.session_state.page == "home":
    st.title("Home Page")
    st.write("Welcome! Use the buttons below to navigate to different pages.")

    # Buttons for navigation
    if st.button("Go to Book Page"):
        navigate_to("book")
    if st.button("Go to Tel Page"):
        navigate_to("tel")

# Redirect to Book Page
elif st.session_state.page == "book":
    # Execute the Book Page logic directly from the `pages/book.py`
    with open("pages/book.py") as f:
        exec(f.read())

# Redirect to Tel Page
elif st.session_state.page == "tel":
    # Execute the Tel Page logic directly from the `pages/tel.py`
    with open("pages/tel.py") as f:
        exec(f.read())
