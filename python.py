# ============================================================
# FACE DETECTION PROJECT USING OPENCV + STREAMLIT
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Important Libraries
# ------------------------------------------------------------

import cv2
import numpy as np
import streamlit as st


# ------------------------------------------------------------
# Streamlit Page
# ------------------------------------------------------------

st.title("Passport Photo Face Detection")
st.write("Upload a passport-size photo to perform OpenCV operations.")


# ------------------------------------------------------------
# STEP 2: Upload Image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your passport-size photo",
    type=["jpg", "jpeg", "png"]
)

# Continue only when the user uploads an image
if uploaded_file is not None:

    # Get uploaded filename
    filename = uploaded_file.name

    st.success("File uploaded successfully!")
    st.write("Uploaded file:", filename)


    # --------------------------------------------------------
    # STEP 3: Read and Display Original Image
    # --------------------------------------------------------

    # Read uploaded file as bytes
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    # Decode bytes into an OpenCV image
    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Check whether image was decoded successfully
    if img is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    # OpenCV uses BGR, while Streamlit displays RGB
    image_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    st.header("Step 3: Original Image")

    st.image(
        image_rgb,
        caption="Original Image"
    )


    # --------------------------------------------------------
    # STEP 4: Image Transformations
    # --------------------------------------------------------

    st.header("Step 4: Image Transformations")

    # Convert image to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Resize image to 300 × 300
    resized = cv2.resize(
        img,
        (300, 300)
    )

    # Rotate image 90 degrees clockwise
    rotated = cv2.rotate(
        img,
        cv2.ROTATE_90_CLOCKWISE
    )

    # Convert BGR images to RGB
    resized_rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    rotated_rgb = cv2.cvtColor(
        rotated,
        cv2.COLOR_BGR2RGB
    )

    # Create three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            gray,
            caption="Grayscale"
        )

    with col2:
        st.image(
            resized_rgb,
            caption="Resized 300 × 300"
        )

    with col3:
        st.image(
            rotated_rgb,
            caption="Rotated 90°"
        )


    # --------------------------------------------------------
    # STEP 5: Drawing Shapes and Adding Text
    # --------------------------------------------------------

    st.header("Step 5: Drawing Shapes and Text")

    # Make copy so original image remains unchanged
    img_copy = img.copy()

    # Get image dimensions
    height, width = img_copy.shape[:2]

    # Draw green rectangle
    cv2.rectangle(
        img_copy,
        (50, 50),
        (min(200, width - 1), min(200, height - 1)),
        (0, 255, 0),
        3
    )

    # Draw blue circle only if image is large enough
    center_x = min(300, width // 2)
    center_y = min(300, height // 2)

    cv2.circle(
        img_copy,
        (center_x, center_y),
        50,
        (255, 0, 0),
        3
    )

    # Add red text
    cv2.putText(
        img_copy,
        "OpenCV Demo",
        (20, min(400, height - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Convert BGR → RGB
    annotated_rgb = cv2.cvtColor(
        img_copy,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        annotated_rgb,
        caption="Annotated Image"
    )


    # --------------------------------------------------------
    # STEP 6: Image Filtering
    # Gaussian Blur + Canny Edge Detection
    # --------------------------------------------------------

    st.header("Step 6: Image Filtering")

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(
        img,
        (15, 15),
        0
    )

    # Convert blurred image to grayscale
    gray_img = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    # Apply Canny Edge Detection
    edges = cv2.Canny(
        gray_img,
        100,
        200
    )

    # Convert blurred image to RGB
    blurred_rgb = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            blurred_rgb,
            caption="Gaussian Blurred Image"
        )

    with col2:
        st.image(
            edges,
            caption="Canny Edge Detection"
        )


    # --------------------------------------------------------
    # STEP 7: Face Detection using Haar Cascade
    # --------------------------------------------------------

    st.header("Step 7: Face Detection")

    # Load pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Convert original image to grayscale
    gray_face = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_face,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Copy original image
    img_faces = img.copy()

    # Draw rectangle around every detected face
    for (x, y, w, h) in faces:

        cv2.rectangle(
            img_faces,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Convert BGR → RGB
    img_faces_rgb = cv2.cvtColor(
        img_faces,
        cv2.COLOR_BGR2RGB
    )

    # Display result
    st.image(
        img_faces_rgb,
        caption=f"Detected Faces: {len(faces)}"
    )

    # Display number of detected faces
    if len(faces) > 0:
        st.success(
            f"{len(faces)} face(s) detected successfully!"
        )
    else:
        st.warning("No face detected.")