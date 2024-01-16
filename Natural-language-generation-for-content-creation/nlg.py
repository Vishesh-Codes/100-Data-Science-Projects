import streamlit as st
import random
import nltk

nltk.download("punkt")

# Larger sample data for content generation
sample_data = {
    "intros": [
        "In today's rapidly evolving landscape,",
        "As we navigate the challenges of the 21st century,",
        "The ever-expanding realm of technology introduces us to",
        "Amidst the global changes,",
    ],
    "topics": [
        "artificial intelligence",
        "blockchain technology",
        "climate change",
        "space exploration",
        "biotechnology",
        "augmented reality",
        "nanotechnology",
        "robotics",
        "renewable energy",
        "genetic engineering",
    ],
    "conclusions": [
        "These transformative developments drive innovation and redefine our future.",
        "It is crucial for us to embrace these advancements for a sustainable tomorrow.",
        "As we witness the convergence of science and technology, the possibilities are limitless.",
        "In the grand tapestry of progress, these breakthroughs weave a promising narrative.",
        "The journey into the future is illuminated by the profound impact of these technological marvels.",
    ],
}

def generate_content():
    intro = random.choice(sample_data["intros"])
    topic = random.choice(sample_data["topics"])
    conclusion = random.choice(sample_data["conclusions"])

    content = f"{intro} {topic}. {conclusion}"
    return content

def main():
    st.title("Content Generation App")

    num_samples = st.slider("Select the number of content pieces to generate", 1, 20, 5)

    if st.button("Generate Content"):
        generated_content = [generate_content() for _ in range(num_samples)]

        for i, content in enumerate(generated_content, start=1):
            st.markdown(f"**Generated Content {i}:**\n{content}\n", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
