import joblib


def select_best_model(results, models):

    # Select model with lowest MAE
    best_model_name = min(results, key=results.get)

    print("\nModel Performance:")
    for model_name, score in results.items():
        print(f"{model_name}: {score}")

    print(f"\nBest Model: {best_model_name}")

    # Get best model object
    best_model = models[best_model_name]

    # Save best model
    joblib.dump(
        best_model,
        "saved_models/best_model.pkl"
    )

    print("\nBest model saved successfully!")

    return best_model_name