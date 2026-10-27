
import tensorflow as tf

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import MobileNetV2, ResNet50


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SHAPE = (224, 224, 3)


def build_classifier(
    model_name="mobilenetv2",
    dropout_rate=0.3,
    l2_value=0.0,
):
    """
    Build a transfer learning classifier.

    dropout_rate:
        0.0 means no Dropout
        0.4 means 40% Dropout

    l2_value:
        0.0 means no L2
        0.001 enables L2 regularization
    """

    if not 0 <= dropout_rate < 1:
        raise ValueError("Invalid dropout_rate")

    if l2_value < 0:
        raise ValueError("l2_value cannot be negative")

    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        backbone = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=IMAGE_SHAPE,
        )

    elif model_name == "resnet50":
        backbone = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=IMAGE_SHAPE,
        )

    else:
        raise ValueError(
            f"Unknown architecture: {model_name}"
        )

    backbone.trainable = False

    inputs = layers.Input(
        shape=IMAGE_SHAPE,
        name="mri_image",
    )

    # Keep pretrained BatchNormalization
    # layers in inference mode.
    x = backbone(inputs, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(
            l2_value
        ),
        name="dense_features",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=regularizers.l2(
            l2_value
        ),
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model_name}_mri_classifier",
    )

    return model


def compile_classifier(
    model,
    learning_rate=0.001,
):
    """
    Compile for four-class classification.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_backbone(model):
    """
    Find the pretrained CNN backbone.
    """

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError("Backbone not found")


def unfreeze_last_layers(
    model,
    number_of_layers=30,
):
    """
    Unfreeze the final backbone layers
    while keeping BatchNormalization frozen.
    """

    if number_of_layers <= 0:
        raise ValueError(
            "number_of_layers must be positive"
        )

    backbone = get_backbone(model)
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-number_of_layers:]:
        if not isinstance(
            layer,
            layers.BatchNormalization,
        ):
            layer.trainable = True

    count = sum(
        int(layer.trainable)
        for layer in backbone.layers
    )

    print("Trainable backbone layers:", count)

    return model


if __name__ == "__main__":

    model = build_classifier(
        model_name="mobilenetv2",
        dropout_rate=0.4,
        l2_value=0.001,
    )

    compile_classifier(model)

    model.summary()
