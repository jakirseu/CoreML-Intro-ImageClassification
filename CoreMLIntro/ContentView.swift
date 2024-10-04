import SwiftUI
import CoreML
import Vision
 

struct ContentView: View {
    @State private var image: UIImage? = nil
    @State private var classificationLabel: String = "Pick an image to classify"
    @State private var isImagePickerPresented: Bool = false
    
    var body: some View {
        VStack {
            Text(classificationLabel)
                .font(.title)
                .padding()
            
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .padding()
            }
            
            Button("Pick an Image") {
                isImagePickerPresented = true
            }
            .padding()
        }
        .sheet(isPresented: $isImagePickerPresented, content: {
            ImagePicker(image: $image, onImagePicked: classifyImage)
        })
    }
    
    // Function to classify the image using Core ML
    func classifyImage(_ image: UIImage) {
        guard let ciImage = CIImage(image: image) else {
            classificationLabel = "Could not convert UIImage to CIImage."
            return
        }
        
        do {
            // Initialize the model with a configuration
            let modelConfiguration = MLModelConfiguration()
            let model = try VNCoreMLModel(for: MobileNetV2(configuration: modelConfiguration).model)
            
            // Create a request for image classification
            let request = VNCoreMLRequest(model: model) { (request, error) in
                if let results = request.results as? [VNClassificationObservation], let topResult = results.first {
                    
                    self.classificationLabel = "Classification: \(topResult.identifier) (\(Int(topResult.confidence * 100))%)"
                    
                } else {
                    
                    self.classificationLabel = "Could not classify image."
                    
                }
            }
            
            // Perform the request
            let handler = VNImageRequestHandler(ciImage: ciImage)
            
            try? handler.perform([request])
            
        } catch {
            classificationLabel = "Could not load model: \(error.localizedDescription)"
        }
    }
}

// ImagePicker helper to allow the user to pick an image
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var onImagePicked: (UIImage) -> Void
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
                parent.onImagePicked(uiImage)
            }
            
            picker.dismiss(animated: true)
        }
    }
}

#Preview {
    ContentView()
}
