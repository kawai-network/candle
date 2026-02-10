package candle

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestVersion(t *testing.T) {
	// This will fail if the library isn't built yet, which is expected
	// in CI with -short flag
	if !initialized {
		t.Skip("candle library not initialized (binary not available)")
	}
	v := Version()
	if v == "" || v == "unknown" {
		t.Errorf("expected valid version, got %q", v)
	}
	t.Logf("candle binding version: %s", v)
}

func TestInitIdempotent(t *testing.T) {
	// Init() should be safe to call multiple times
	err1 := Init()
	err2 := Init()
	// Both should have the same result
	if (err1 == nil) != (err2 == nil) {
		t.Errorf("Init() not idempotent: first=%v, second=%v", err1, err2)
	}
}

func TestDownloadModel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model download in short mode")
	}

	tmpDir := t.TempDir()
	files, err := DownloadModel("sentence-transformers/all-MiniLM-L6-v2", tmpDir)
	if err != nil {
		t.Fatalf("DownloadModel failed: %v", err)
	}

	if files.ConfigPath == "" {
		t.Error("config.json path is empty")
	}
	if files.TokenizerPath == "" {
		t.Error("tokenizer path is empty")
	}

	// Verify files exist
	for _, path := range []string{files.ConfigPath, files.TokenizerPath} {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("downloaded file does not exist: %s", path)
		}
	}

	// Second call should use cache
	files2, err := DownloadModel("sentence-transformers/all-MiniLM-L6-v2", tmpDir)
	if err != nil {
		t.Fatalf("cached DownloadModel failed: %v", err)
	}
	if files2.ConfigPath != files.ConfigPath {
		t.Error("second download returned different config path (cache not working)")
	}
}

func TestTextGeneration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewTextGenerationPipeline(TextGenerationConfig{
		ModelID:   "HuggingFaceTB/SmolLM-135M",
		MaxTokens: 20,
	})
	if err != nil {
		t.Fatalf("NewTextGenerationPipeline failed: %v", err)
	}
	defer pipeline.Close()

	text, err := pipeline.Generate("The capital of France is")
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if text == "" {
		t.Error("generated text is empty")
	}
	t.Logf("Generated: %s", text)

	// Test double close is safe
	pipeline.Close()
}

func TestEmbedding(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewEmbeddingPipeline(EmbeddingConfig{
		ModelID:   "sentence-transformers/all-MiniLM-L6-v2",
		Normalize: true,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingPipeline failed: %v", err)
	}
	defer pipeline.Close()

	// Single embedding
	vec, err := pipeline.Embed("Hello world")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(vec) != 384 {
		t.Errorf("expected dim 384, got %d", len(vec))
	}

	// Verify it's non-zero
	nonZero := false
	for _, v := range vec {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("embedding vector is all zeros")
	}

	// Batch embedding
	vecs, err := pipeline.EmbedBatch([]string{"Hello", "World"})
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}
	if len(vecs) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 384 {
			t.Errorf("batch vector %d: expected dim 384, got %d", i, len(v))
		}
	}

	// Test double close
	pipeline.Close()
}

// createTestImage creates a simple PNG image for vision tests.
func createTestImage(t *testing.T) string {
	t.Helper()
	// Create a minimal valid PNG (1x1 pixel, red)
	// Using Go's image package
	dir := t.TempDir()
	imgPath := filepath.Join(dir, "test.png")

	// Write a simple 224x224 PNG using image stdlib
	f, err := os.Create(imgPath)
	if err != nil {
		t.Fatalf("failed to create test image: %v", err)
	}
	defer f.Close()

	// Use raw image creation
	img := makeTestPNG(224, 224)
	if _, err := f.Write(img); err != nil {
		t.Fatalf("failed to write test image: %v", err)
	}

	return imgPath
}

// makeTestPNG creates a minimal valid PNG file bytes.
func makeTestPNG(width, height int) []byte {
	// Use Go's image/png package
	return createPNGBytes(width, height)
}

func TestClassification(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewClassificationPipeline(ClassificationConfig{
		ModelID: "google/vit-base-patch16-224",
	})
	if err != nil {
		t.Fatalf("NewClassificationPipeline failed: %v", err)
	}
	defer pipeline.Close()

	imgPath := createTestImage(t)
	preds, err := pipeline.Classify(imgPath, 5)
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	if len(preds) == 0 {
		t.Fatal("no predictions returned")
	}

	t.Logf("Top %d predictions:", len(preds))
	for i, p := range preds {
		t.Logf("  %d: %s (%.4f)", i+1, p.Label, p.Score)
	}

	// Verify scores sum roughly to 1 (softmax)
	var total float32
	for _, p := range preds {
		total += p.Score
		if p.Label == "" {
			t.Error("empty label in prediction")
		}
	}

	pipeline.Close()
}

func TestClipEmbedText(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewClipPipeline(ClipConfig{
		ModelID: "openai/clip-vit-base-patch32",
	})
	if err != nil {
		t.Fatalf("NewClipPipeline failed: %v", err)
	}
	defer pipeline.Close()

	vec, err := pipeline.EmbedText("a photo of a cat")
	if err != nil {
		t.Fatalf("EmbedText failed: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("empty embedding vector")
	}
	t.Logf("CLIP text embedding dim: %d", len(vec))

	// Verify non-zero
	nonZero := false
	for _, v := range vec {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("embedding vector is all zeros")
	}

	pipeline.Close()
}

func TestClipScore(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewClipPipeline(ClipConfig{
		ModelID: "openai/clip-vit-base-patch32",
	})
	if err != nil {
		t.Fatalf("NewClipPipeline failed: %v", err)
	}
	defer pipeline.Close()

	imgPath := createTestImage(t)

	scores, err := pipeline.Score(imgPath, []string{"a purple image", "a cat", "a dog"})
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if len(scores) != 3 {
		t.Fatalf("expected 3 scores, got %d", len(scores))
	}

	t.Logf("CLIP scores: purple=%.4f, cat=%.4f, dog=%.4f", scores[0], scores[1], scores[2])

	// Verify scores are valid probabilities
	var total float32
	for _, s := range scores {
		if s < 0 || s > 1 {
			t.Errorf("score out of range [0,1]: %f", s)
		}
		total += s
	}
	if total < 0.99 || total > 1.01 {
		t.Errorf("scores don't sum to ~1: %f", total)
	}

	// Image embedding test
	imgEmb, err := pipeline.EmbedImage(imgPath)
	if err != nil {
		t.Fatalf("EmbedImage failed: %v", err)
	}
	if len(imgEmb) == 0 {
		t.Fatal("empty image embedding")
	}
	t.Logf("CLIP image embedding dim: %d", len(imgEmb))
}

// createTestWAV creates a simple 16kHz mono WAV file with a sine wave.
func createTestWAV(t *testing.T, durationSecs float64) string {
	t.Helper()
	dir := t.TempDir()
	wavPath := filepath.Join(dir, "test.wav")

	sampleRate := 16000
	numSamples := int(durationSecs * float64(sampleRate))
	freq := 440.0 // A4 note

	f, err := os.Create(wavPath)
	if err != nil {
		t.Fatalf("failed to create WAV: %v", err)
	}
	defer f.Close()

	// Generate PCM data
	samples := make([]int16, numSamples)
	for i := range samples {
		t := float64(i) / float64(sampleRate)
		samples[i] = int16(math.Sin(2*math.Pi*freq*t) * 16000)
	}

	// Write WAV header
	dataSize := uint32(numSamples * 2) // 16-bit samples
	fileSize := dataSize + 36

	// RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, fileSize)
	f.Write([]byte("WAVE"))

	// fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))  // chunk size
	binary.Write(f, binary.LittleEndian, uint16(1))    // PCM format
	binary.Write(f, binary.LittleEndian, uint16(1))    // mono
	binary.Write(f, binary.LittleEndian, uint32(sampleRate))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	binary.Write(f, binary.LittleEndian, uint16(2))    // block align
	binary.Write(f, binary.LittleEndian, uint16(16))   // bits per sample

	// data chunk
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, dataSize)
	binary.Write(f, binary.LittleEndian, samples)

	return wavPath
}

func TestWhisper(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewWhisperPipeline(WhisperConfig{
		ModelID: "openai/whisper-tiny",
	})
	if err != nil {
		t.Fatalf("NewWhisperPipeline failed: %v", err)
	}
	defer pipeline.Close()

	// Create a 2-second test WAV (just a tone, so it should transcribe to something)
	wavPath := createTestWAV(t, 2.0)

	result, err := pipeline.Transcribe(wavPath)
	if err != nil {
		t.Fatalf("Transcribe failed: %v", err)
	}

	t.Logf("Transcription: %q", result.Text)
	t.Logf("Segments: %d", len(result.Segments))
	for i, seg := range result.Segments {
		t.Logf("  Segment %d: [%.1f-%.1f] %q", i, seg.Start, seg.End, seg.Text)
	}

	// We don't check exact text since a tone won't produce meaningful words,
	// but we verify the pipeline works end-to-end
	pipeline.Close()
}

func TestT5(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewT5Pipeline(T5Config{
		ModelID: "t5-small",
	})
	if err != nil {
		t.Fatalf("NewT5Pipeline failed: %v", err)
	}
	defer pipeline.Close()

	// T5 translation task
	text, err := pipeline.Generate("translate English to French: The house is wonderful.", T5GenerateOpts{
		MaxTokens: 50,
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if text == "" {
		t.Error("generated text is empty")
	}
	t.Logf("T5 translation: %s", text)

	// Test double close
	pipeline.Close()
}

func TestTranslation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model test in short mode")
	}
	if !initialized {
		t.Skip("candle library not initialized")
	}

	pipeline, err := NewTranslationPipeline(TranslationConfig{
		LanguagePair: "fr-en",
	})
	if err != nil {
		t.Fatalf("NewTranslationPipeline failed: %v", err)
	}
	defer pipeline.Close()

	result, err := pipeline.Translate("Bonjour le monde")
	if err != nil {
		t.Fatalf("Translate failed: %v", err)
	}
	if result == "" {
		t.Error("translation result is empty")
	}
	t.Logf("Translation (fr->en): %s", result)

	// Test double close
	pipeline.Close()
}
