"""
Model factory - creates all Keras models with transfer learning
Project: Plant Disease Detection and Crop Yield Prediction
Dataset: Bangladeshi Crops Disease Dataset (15 classes)

Model Assignments:
- Zahra: 13 models (Xception, VGG16, ResNet152/V2, MobileNet/V2, EfficientNetB0-B2, EfficientNetV2B0/S/M, ConvNeXtLarge)
- Sireen: 12 models (VGG19, ResNet50/V2, DenseNet121/169/201, EfficientNetB3-B5, EfficientNetV2L, ConvNeXtTiny/Small)
- Tala: 13 models (ResNet101/V2, InceptionV3, InceptionResNetV2, NASNetMobile/Large, EfficientNetB6-B7, EfficientNetV2B1-B3, ConvNeXtBase/XLarge)
"""
import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =============================================================================
# USER MODEL ASSIGNMENTS - Use these lists to filter models for each user
# =============================================================================

ZAHRA_MODELS = [
    'Xception', 'VGG16', 'ResNet152', 'ResNet152V2', 'MobileNet', 'MobileNetV2',
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetV2B0',
    'EfficientNetV2S', 'EfficientNetV2M', 'ConvNeXtLarge'
]

SIREEN_MODELS = [
    'VGG19', 'ResNet50', 'ResNet50V2', 'DenseNet121', 'DenseNet169', 'DenseNet201',
    'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetV2L',
    'ConvNeXtTiny', 'ConvNeXtSmall'
]

TALA_MODELS = [
    'ResNet101', 'ResNet101V2', 'InceptionV3', 'InceptionResNetV2',
    'NASNetMobile', 'NASNetLarge', 'EfficientNetB6', 'EfficientNetB7',
    'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
    'ConvNeXtBase', 'ConvNeXtXLarge'
]

# All models combined (for reference)
ALL_MODELS = ZAHRA_MODELS + SIREEN_MODELS + TALA_MODELS

# =============================================================================
# MODEL CONFIGURATION - Input sizes and preprocessing functions
# =============================================================================

MODEL_CONFIG = {
    # ===================== ZAHRA'S MODELS =====================
    'Xception': {
        'input_size': (299, 299), 
        'preprocess': tf.keras.applications.xception.preprocess_input,
        'owner': 'Zahra'
    },
    'VGG16': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.vgg16.preprocess_input,
        'owner': 'Zahra'
    },
    'ResNet152': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'owner': 'Zahra'
    },
    'ResNet152V2': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet_v2.preprocess_input,
        'owner': 'Zahra'
    },
    'MobileNet': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.mobilenet.preprocess_input,
        'owner': 'Zahra'
    },
    'MobileNetV2': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetB0': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetB1': {
        'input_size': (240, 240), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetB2': {
        'input_size': (260, 260), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetV2B0': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetV2S': {
        'input_size': (384, 384), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Zahra'
    },
    'EfficientNetV2M': {
        'input_size': (480, 480), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Zahra'
    },
    'ConvNeXtLarge': {
        'input_size': (224, 224), 
        'preprocess': None,
        'owner': 'Zahra'
    },

    # ===================== SIREEN'S MODELS =====================
    'VGG19': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
        'owner': 'Sireen'
    },
    'ResNet50': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'owner': 'Sireen'
    },
    'ResNet50V2': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet_v2.preprocess_input,
        'owner': 'Sireen'
    },
    'DenseNet121': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'owner': 'Sireen'
    },
    'DenseNet169': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'owner': 'Sireen'
    },
    'DenseNet201': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'owner': 'Sireen'
    },
    'EfficientNetB3': {
        'input_size': (300, 300), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Sireen'
    },
    'EfficientNetB4': {
        'input_size': (380, 380), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Sireen'
    },
    'EfficientNetB5': {
        'input_size': (456, 456), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Sireen'
    },
    'EfficientNetV2L': {
        'input_size': (480, 480), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Sireen'
    },
    'ConvNeXtTiny': {
        'input_size': (224, 224), 
        'preprocess': None,
        'owner': 'Sireen'
    },
    'ConvNeXtSmall': {
        'input_size': (224, 224), 
        'preprocess': None,
        'owner': 'Sireen'
    },

    # ===================== TALA'S MODELS =====================
    'ResNet101': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'owner': 'Tala'
    },
    'ResNet101V2': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.resnet_v2.preprocess_input,
        'owner': 'Tala'
    },
    'InceptionV3': {
        'input_size': (299, 299), 
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
        'owner': 'Tala'
    },
    'InceptionResNetV2': {
        'input_size': (299, 299), 
        'preprocess': tf.keras.applications.inception_resnet_v2.preprocess_input,
        'owner': 'Tala'
    },
    'NASNetMobile': {
        'input_size': (224, 224), 
        'preprocess': tf.keras.applications.nasnet.preprocess_input,
        'owner': 'Tala'
    },
    'NASNetLarge': {
        'input_size': (331, 331), 
        'preprocess': tf.keras.applications.nasnet.preprocess_input,
        'owner': 'Tala'
    },
    'EfficientNetB6': {
        'input_size': (528, 528), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Tala'
    },
    'EfficientNetB7': {
        'input_size': (600, 600), 
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'owner': 'Tala'
    },
    'EfficientNetV2B1': {
        'input_size': (240, 240), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Tala'
    },
    'EfficientNetV2B2': {
        'input_size': (260, 260), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Tala'
    },
    'EfficientNetV2B3': {
        'input_size': (300, 300), 
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'owner': 'Tala'
    },
    'ConvNeXtBase': {
        'input_size': (224, 224), 
        'preprocess': None,
        'owner': 'Tala'
    },
    'ConvNeXtXLarge': {
        'input_size': (224, 224), 
        'preprocess': None,
        'owner': 'Tala'
    },
}


def get_model(model_name, num_classes=15, learning_rate=0.0001):
    """
    Create a model with transfer learning

    Args:
        model_name: Name of the model (must be in MODEL_CONFIG)
        num_classes: Number of output classes (default: 15 for Bangladeshi Crops Disease)
        learning_rate: Learning rate for optimizer (default: 0.0001)

    Returns:
        model: Compiled Keras model
        preprocess_func: Preprocessing function for the model
        input_size: Input image size (height, width)
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[model_name]
    input_size = config['input_size']
    preprocess_func = config['preprocess']
    owner = config['owner']

    print(f"Creating {model_name} (Owner: {owner}) with input size {input_size}")

    # Get the base model class from tensorflow.keras.applications
    model_class = globals()[model_name]

    # Create base model with ImageNet weights
    base_model = model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(*input_size, 3)
    )

    # Freeze base layers for transfer learning
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output layer with float32 (required for mixed precision training)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model, preprocess_func, input_size


def unfreeze_layers(model, num_layers_to_unfreeze=10):
    """
    Unfreeze top layers for fine-tuning

    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the top (default: 10)

    Returns:
        model: Recompiled model with unfrozen layers
    """
    # Find the base model (first layer that is a Model instance)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model):
            base_model = layer
            break

    if base_model is None:
        print("No base model found")
        return model

    # Unfreeze last N layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    print(f"Unfroze last {num_layers_to_unfreeze} layers of {base_model.name}")

    return model


def get_user_models(user_name):
    """
    Get list of models assigned to a specific user

    Args:
        user_name: One of 'Zahra', 'Sireen', or 'Tala'

    Returns:
        list: List of model names assigned to the user
    """
    user_name = user_name.capitalize()
    if user_name == 'Zahra':
        return ZAHRA_MODELS
    elif user_name == 'Sireen':
        return SIREEN_MODELS
    elif user_name == 'Tala':
        return TALA_MODELS
    else:
        raise ValueError(f"User {user_name} not found. Choose from: Zahra, Sireen, Tala")


def print_model_assignments():
    """Print a formatted table of all model assignments"""
    print("\n" + "="*80)
    print("MODEL ASSIGNMENTS - Plant Disease Detection Project")
    print("="*80)

    print(f"\n🌿 ZAHRA ({len(ZAHRA_MODELS)} models):")
    for i, model in enumerate(ZAHRA_MODELS, 1):
        size = MODEL_CONFIG[model]['input_size']
        print(f"   {i:2d}. {model:<20} ({size[0]}×{size[1]})")

    print(f"\n🌿 SIREEN ({len(SIREEN_MODELS)} models):")
    for i, model in enumerate(SIREEN_MODELS, 1):
        size = MODEL_CONFIG[model]['input_size']
        print(f"   {i:2d}. {model:<20} ({size[0]}×{size[1]})")

    print(f"\n🌿 TALA ({len(TALA_MODELS)} models):")
    for i, model in enumerate(TALA_MODELS, 1):
        size = MODEL_CONFIG[model]['input_size']
        print(f"   {i:2d}. {model:<20} ({size[0]}×{size[1]})")

    print("\n" + "="*80)
    print(f"Total: {len(ALL_MODELS)} models")
    print("="*80)


# Print assignments when module is loaded
if __name__ == "__main__":
    print_model_assignments()