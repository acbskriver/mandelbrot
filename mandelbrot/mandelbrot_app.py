import streamlit as st
from mandelbrot.mandelbrot_functions import mandelbrot_dataset_creator, initialize_model
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

WIDTH = 1300
HEIGHT = 1100

st.set_page_config(layout="wide")


st.sidebar.title("Mandelbrot parameters")
real_midpoint_input = st.sidebar.slider(
    "real part midpoint", min_value=-2.25, max_value=0.75, value=-0.75, step=0.01
)
imag_midpoint_input = st.sidebar.slider(
    "imaginary part midpoint", min_value=-1.1, max_value=1.1, value=0.0, step=0.01
)
zoom = st.sidebar.slider(
    "zoom factor", min_value=1.0, max_value=100.0, value=1.0, step=0.1
)

zoom2 = st.sidebar.slider(
    "zoom factor 2", min_value=1.0, max_value=100.0, value=1.0, step=0.1
)

exponent = st.sidebar.slider(
    "exponent", min_value=1.0, max_value=25.0, value=2.0, step=0.01
)

exponent2 = st.sidebar.slider(
    "exponent 2", min_value=1.0, max_value=25.0, value=1.0, step=0.01
)

nr_iterations = st.sidebar.slider(
    "nr_iterations", min_value=10, max_value=1000, value=100, step=1
)

classification_threshold = st.sidebar.slider(
    "classification_threshold", min_value=1.0, max_value=10.0, value=1.0, step=0.01
)


st.title("Neural Networks as universal function approximators")

df = mandelbrot_dataset_creator(
    real_start=real_midpoint_input - (1.5 / (zoom * zoom2)),
    real_end=real_midpoint_input + (1.5 / (zoom * zoom2)),
    imag_start=imag_midpoint_input * 1j - (1.1j / (zoom * zoom2)),
    imag_end=imag_midpoint_input * 1j + (1.1j / (zoom * zoom2)),
    real_nr_pixels=int(WIDTH / 2),
    imag_nr_pixels=int(HEIGHT / 2),
    exponent=exponent * exponent2,
    nr_iterations=nr_iterations,
    classification_threshold=classification_threshold,
)

fig = px.imshow(df, color_continuous_scale="Viridis")
fig.update_layout(width=WIDTH, height=HEIGHT)
fig.update_layout(coloraxis_showscale=False)

st.plotly_chart(fig)

col1, col2, col3 = st.columns([4, 1, 2])
with col1:
    epoch_nr = st.slider("epoch_nr", min_value=1, max_value=100, value=10, step=1)
    es_patience = st.slider("es_patience", min_value=1, max_value=25, value=5, step=1)
    red_lr_patience = st.slider(
        "red_lr_patience", min_value=1, max_value=10, value=3, step=1
    )

with col3:
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    button = st.button("Approximate with NN")
    text_placeholder = st.empty()


if button:
    # get dimension for later usage
    c_real = np.float64(df.columns)
    c_imag = df.index

    # transpose/ transform data from shape (1000, 1000) -> (1000000, 3)
    data = pd.melt(df.reset_index(), id_vars="index", value_vars=df.columns[1:]).astype(
        {"variable": float}
    )

    # rename cols
    data.columns = ["imag_part", "real_part", "diverges_at_iter"]

    # create features, targets
    X = data[["real_part", "imag_part"]]
    y = data[["diverges_at_iter"]]

    # create preprocessor pipe
    # X_preprocessor_pipe = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=2), MinMaxScaler())
    y_preprocessor_pipe = make_pipeline(MinMaxScaler())
    X_preprocessor_pipe = make_pipeline(MinMaxScaler())

    # standardscale all data
    X_trans = X_preprocessor_pipe.fit_transform(X)
    y_trans = y_preprocessor_pipe.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y_trans)

    model = initialize_model()

    # define callbacks
    early_stopping = EarlyStopping(
        patience=es_patience, monitor="val_loss", restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=red_lr_patience, min_lr=0.00001
    )

    class PrintAtEpochEnd(Callback):
        def __init__(self):
            super(PrintAtEpochEnd, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            text_placeholder.text(f"   Epoch {epoch + 1} finished!")

    print_at_epoch_end = PrintAtEpochEnd()

    # train model (always with ca. batch_size = 1000)
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=epoch_nr,
        callbacks=[early_stopping, reduce_lr, print_at_epoch_end],
    )

    predictions = model.predict(X_trans, batch_size=int(len(X_train) / 100))

    # inverse transform predicitions
    predictions = y_preprocessor_pipe.inverse_transform(predictions)

    # load predictions into df and convert to grid
    prediction_df = pd.DataFrame(
        {
            "real_part": X["real_part"],
            "imag_part": X["imag_part"],
            "prediction": predictions[:, 0],
        }
    )
    prediction_df = prediction_df.pivot(
        index="imag_part", columns="real_part", values="prediction"
    )
    prediction_df.index.name = None
    prediction_df.columns.name = None

    fig_predicted = px.imshow(prediction_df, color_continuous_scale="Viridis")
    fig_predicted.update_layout(width=WIDTH, height=HEIGHT)
    fig_predicted.update_layout(coloraxis_showscale=False)

    st.title("Approximated Mandelbrot set")

    st.plotly_chart(fig_predicted)
