from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ustawienie zmiennej środowiskowej, aby wyłączyć ostrzeżenie o symlinkach
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Wybór modelu
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Ładowanie tokenizera i modelu
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)


def przygotuj_prompt(pytanie, historia_rozmowy=None):
    if historia_rozmowy is None:
        historia_rozmowy = []

    prompt = "<|system|>\nJesteś pomocnym, przyjaznym i dokładnym asystentem.\n"

    for wiadomosc in historia_rozmowy:
        if wiadomosc["rola"] == "user":
            prompt += f"<|user|>\n{wiadomosc['tresc']}\n"
        else:  # assistant
            prompt += f"<|assistant|>\n{wiadomosc['tresc']}\n"

    prompt += f"<|user|>\n{pytanie}\n<|assistant|>\n"
    return prompt


def generuj_odpowiedz(pytanie, historia_rozmowy=None):
    prompt = przygotuj_prompt(pytanie, historia_rozmowy)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    pelna_odpowiedz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    odpowiedz = pelna_odpowiedz.split("<|assistant|>")[-1].strip()

    return odpowiedz


def main():
    print("Witaj! Jestem asystentem opartym na modelu LLM.")
    print("Możesz ze mną porozmawiać. Wpisz 'koniec' aby zakończyć.")

    historia_rozmowy = []

    while True:
        pytanie = input("\nTy: ")
        if pytanie.lower() in ["koniec", "exit", "quit"]:
            print("Do widzenia!")
            break

        odpowiedz = generuj_odpowiedz(pytanie, historia_rozmowy)
        print(f"\nAsystent: {odpowiedz}")

        historia_rozmowy.append({"rola": "user", "tresc": pytanie})
        historia_rozmowy.append({"rola": "assistant", "tresc": odpowiedz})

        if len(historia_rozmowy) > 10:
            historia_rozmowy = historia_rozmowy[-10:]


if __name__ == "__main__":
    main()
