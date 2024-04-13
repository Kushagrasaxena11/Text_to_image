import streamlit as st
from diffusers.pipelines import VQDiffusionPipeline  # Update import statement

def main():
    # Load the DiffusionPipeline model
    pipe = VQDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

    # Streamlit app title and description
    st.title("Image Generation with Diffusion Models")
    st.write("This app generates images based on prompts using a Diffusion model.")

    # Define a text input for user prompt
    prompt = st.text_input("Enter your prompt", "An astronaut riding a green horse")

    if st.button("Generate Image"):
        # Generate image based on user prompt
        with st.spinner("Generating..."):
            images = pipe(prompt=prompt).images[0]

        # Display generated image
        st.image(images, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
