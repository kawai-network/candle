package candle

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
)

// createPNGBytes creates a simple solid-color PNG image.
func createPNGBytes(width, height int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Fill with a solid color
	c := color.RGBA{R: 128, G: 64, B: 200, A: 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, c)
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic("failed to encode PNG: " + err.Error())
	}
	return buf.Bytes()
}
