import streamlit as st

PAGE_SEQUENCE = [
    {"name": "1. Homepage", "path": "pages/1_Homepage.py"},
    {"name": "2. Team Presentation", "path": "pages/2_Team_Presentation.py"},
    {"name": "3. Flowcharts", "path": "pages/3_Flowcharts.py"},
    {"name": "4. Key Focus Areas", "path": "pages/4_Key_Focus_Areas.py"},
    {"name": "5. FastAPI Demo", "path": "pages/5_FastAPI_Demo.py"},
    {"name": "6. Monitoring and Operations", "path": "pages/6_Monitoring_and_Operations.py"},
    {"name": "7. Business Metrics and Insights", "path": "pages/7_Business_Metrics_and_Insights.py"},
    {"name": "8: Future Improvements", "path": "pages/8_Future_Improvements.py"}
]

def add_pagination_and_footer(current_page_path):
    # Find current page index in sequence
    current_index = next(
        (
            i
            for i, page in enumerate(PAGE_SEQUENCE)
            if page["path"] == current_page_path
        ),
        0,
    )

    # Create columns for previous, current page indicator, next
    prev_butt, next_butt = st.columns(
        2
    )  # elf* and erc* as in "empty left column" and "empty right column"

    # Previous button
    with prev_butt:
        if current_index > 0:  # Not on first page
            prev_page = PAGE_SEQUENCE[current_index - 1]
            if st.button("← Previous", use_container_width=True):
                st.switch_page(prev_page["path"])

    # Next button
    with next_butt:
        if current_index < len(PAGE_SEQUENCE) - 1:  # Not on last page
            next_page = PAGE_SEQUENCE[current_index + 1]
            if st.button("Next →", use_container_width=True):
                st.switch_page(next_page["path"])

    # Copyright line and page indicator
    st.markdown(
        f"© 2025 // Marie Ernø-Møller, Peter Stieg, Qi Bao, Robert Wilson // [Page {current_index + 1}/{len(PAGE_SEQUENCE)}]"
    )

def hw():
    """Test function to print "Hello, world!".

    This function serves as a placeholder to demonstrate the module's structure.
    """
    print("Hello, world!")