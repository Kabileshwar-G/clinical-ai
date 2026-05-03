def generate_explanation(data, prediction):
    """
    Hybrid LLM:
    - Tries HuggingFace locally
    - Falls back in Docker or failure
    """

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "google/flan-t5-small"  # lighter than base

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        prompt = f"""
        Patient data:
        Glucose: {data['Glucose']}
        BMI: {data['BMI']}
        Age: {data['Age']}

        Prediction: {"Diabetic" if prediction == 1 else "Non-diabetic"}

        Explain clearly and give simple medical advice:
        """

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=120)

        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return explanation

    except Exception:
        # 🔥 FALLBACK (DOCKER SAFE)
        if prediction == 1:
            return (
                f"Patient is likely diabetic due to high glucose ({data['Glucose']}) "
                f"and BMI ({data['BMI']}). Recommend healthy diet, exercise, and medical consultation."
            )
        else:
            return (
                "Patient is likely non-diabetic. Maintain a balanced diet and active lifestyle."
            )