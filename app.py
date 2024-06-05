import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import L1, L2, L1L2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import time

# Set page configuration
# st.set_page_config(page_title="TensorFlow Playground", layout="wide")

st.title("🖥️ Tensor Playground")

# Sidebar for user inputs
st.sidebar.header("Model Configuration")

# Set the working directory
os.chdir("C:\\Users\\DELL\\Desktop\\Multiple CSV")

# Select the dataset
data_files = os.listdir()
data = st.sidebar.selectbox("Select the type of dataset", data_files)
current_dir = os.getcwd()
file_path = os.path.join(current_dir, data)

# Load dataset
if file_path:
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    n_classes = len(np.unique(y))

    # Encode labels if there are more than 2 classes
    if n_classes > 2:
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1, 1))

    # Split data
    test_size = st.sidebar.slider("Ratio of training to test data: ", 10, 90) / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Define input dimensions
    input_dim = X.shape[1]
    input_shape = (input_dim,)

    # Model parameters
    active_hidden = st.sidebar.selectbox("Activation Function for Hidden Layers", ("sigmoid", "tanh", "relu"))
    active_output = st.sidebar.selectbox("Activation Function for Output Layer", ("sigmoid", "linear", "softmax"))
    regularizer_option = st.sidebar.selectbox("Regularizer", ("None", "L1", "L2", "Elastic Net"))
    batch_size = st.sidebar.slider("Enter batch size:", 1, 1000, step=1)
    epochs = st.sidebar.number_input("Enter number of epochs:", min_value=1, value=10, step=1)
    hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 10, 1)

    # Initialize session state for hidden layer sizes if not already done
    if 'hidden_size' not in st.session_state:
        st.session_state.hidden_size = [1] * hidden_layers

    # Adjust hidden_size list length based on the number of hidden layers
    if len(st.session_state.hidden_size) != hidden_layers:
        st.session_state.hidden_size = [1] * hidden_layers

    # Set the regularizer
    if regularizer_option == 'L1':
        regularizer = L1(l1=0.01)
    elif regularizer_option == 'L2':
        regularizer = L2(l2=0.01)
    elif regularizer_option == 'Elastic Net':
        regularizer = L1L2(l1=0.01, l2=0.01)
    else:
        regularizer = None

    # Use unique keys for each number_input to ensure state management
    for i in range(hidden_layers):
        st.session_state.hidden_size[i] = st.sidebar.number_input(
            f"Enter number of neurons for hidden layer {i + 1}:",
            min_value=1,
            value=st.session_state.hidden_size[i],
            key=f"hidden_{i+1}"
        )    

    # Train model button
    if st.sidebar.button("Train Model"):
        # Build model
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))

        # Adding hidden layers
        for i in range(hidden_layers):
            model.add(layers.Dense(st.session_state.hidden_size[i], activation=active_hidden, kernel_regularizer=regularizer))

        # Output layer
        output_activation = active_output if n_classes > 2 else 'sigmoid'
        model.add(layers.Dense(n_classes if n_classes > 2 else 1, activation=output_activation))

        # Compile model
        if n_classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'
        
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        # Train model
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        end_time = time.time()
        train_time = end_time - start_time

        st.session_state.model = model
        st.session_state.history = history.history

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Display results
        st.write("### Model Performance")
        st.write(f"Loss: {loss}")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Training Time: {train_time:.2f} seconds")

         # Plot training history
        st.write("### Training History")
        st.line_chart(pd.DataFrame(history.history))


        # Plot decision regions
        st.write("### Decision Regions")
        fig, ax = plt.subplots()
        plot_decision_regions(X_test, np.argmax(y_test, axis=1).astype(int) if n_classes > 2 else y_test.astype(int), clf=model, legend=2)
        st.pyplot(fig)

        # Display model summary
        st.write("### Model Summary")
        model.summary(print_fn=lambda x: st.text(x))
else:
    st.write("Please select a dataset to start.")
