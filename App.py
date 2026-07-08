import streamlit as st
from streamlit_option_menu import option_menu

from Train import DATA_PATH, load_dataset, Train
from Home import Home
from Statistics import Statistics
from Info import Info

class App:
    PAGES = ["Home", "Statistics", "Info"]
    ICONS = ["house", "bar-chart", "people"]
 
    def __init__(self):
        st.set_page_config(
            page_title="Student Performance Predictor",
            page_icon="🎓",
            layout="wide",
        )
        self.df = load_dataset(DATA_PATH)
        self.train = Train()
 
    def _sidebar_nav(self) -> str:
        with st.sidebar:
            st.markdown("## 🎓 Student Predictor")
            selected = option_menu(
                menu_title="Main Menu",
                options=self.PAGES,
                icons=self.ICONS,
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px 0"},
                    "nav-link-selected": {"background-color": "#000080"},
                },
            )
        return selected
 
    def run(self):
        page = self._sidebar_nav()
 
        if page == "Home":
            Home(self.df, self.train).render()
        elif page == "Statistics":
            Statistics(self.df).render()
        elif page == "Info":
            Info().render()
 
 
if __name__ == "__main__":
    App().run()