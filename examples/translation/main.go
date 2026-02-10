package main

import (
	"fmt"
	"log"
	"os"

	"github.com/soundprediction/go-candle/pkg/candle"
)

func main() {
	if err := candle.Init(); err != nil {
		log.Fatalf("Failed to initialize candle: %v", err)
	}
	fmt.Println("candle version:", candle.Version())

	// --- T5 Translation ---
	fmt.Println("\n--- T5 Translation ---")
	t5, err := candle.NewT5Pipeline(candle.T5Config{
		ModelID: "t5-small",
	})
	if err != nil {
		log.Fatalf("T5 pipeline failed: %v", err)
	}
	defer t5.Close()

	phrases := []string{
		"translate English to French: The house is wonderful.",
		"translate English to German: Good morning, how are you?",
		"summarize: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance on tasks through experience.",
	}
	for _, phrase := range phrases {
		result, err := t5.Generate(phrase, candle.T5GenerateOpts{MaxTokens: 64})
		if err != nil {
			log.Printf("T5 error: %v", err)
			continue
		}
		fmt.Printf("  Input:  %s\n  Output: %s\n\n", phrase, result)
	}

	// --- Marian MT Translation ---
	fmt.Println("--- Marian MT (French → English) ---")
	pair := "fr-en"
	if len(os.Args) > 1 {
		pair = os.Args[1]
	}

	mt, err := candle.NewTranslationPipeline(candle.TranslationConfig{
		LanguagePair: pair,
	})
	if err != nil {
		log.Fatalf("Translation pipeline failed: %v", err)
	}
	defer mt.Close()

	inputs := []string{
		"Bonjour le monde",
		"La vie est belle",
		"Je suis un développeur",
	}
	for _, input := range inputs {
		result, err := mt.Translate(input)
		if err != nil {
			log.Printf("Translation error: %v", err)
			continue
		}
		fmt.Printf("  %s → %s\n", input, result)
	}
}
