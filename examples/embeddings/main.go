package main

import (
	"fmt"
	"log"
	"math"

	"github.com/soundprediction/go-candle/pkg/candle"
)

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func main() {
	if err := candle.Init(); err != nil {
		log.Fatalf("Failed to initialize candle: %v", err)
	}
	fmt.Println("candle version:", candle.Version())

	// Create an embedding pipeline with a small sentence-transformer model.
	pipeline, err := candle.NewEmbeddingPipeline(candle.EmbeddingConfig{
		ModelID:   "sentence-transformers/all-MiniLM-L6-v2",
		Normalize: true,
	})
	if err != nil {
		log.Fatalf("Failed to create embedding pipeline: %v", err)
	}
	defer pipeline.Close()

	// Sentences to compare — the first two are semantically similar.
	sentences := []string{
		"The cat sat on the mat",
		"A kitten was resting on a rug",
		"The stock market crashed yesterday",
	}

	// Generate embeddings in a single batch.
	embeddings, err := pipeline.EmbedBatch(sentences)
	if err != nil {
		log.Fatalf("Batch embedding failed: %v", err)
	}

	fmt.Printf("\nEmbedding dimension: %d\n", len(embeddings[0]))

	// Print pairwise cosine similarities.
	fmt.Println("\nPairwise cosine similarities:")
	for i := 0; i < len(sentences); i++ {
		for j := i + 1; j < len(sentences); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("  [%d] vs [%d]: %.4f\n", i, j, sim)
			fmt.Printf("       %q\n       %q\n\n", sentences[i], sentences[j])
		}
	}
}
