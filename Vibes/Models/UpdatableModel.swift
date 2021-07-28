/// Copyright (c) 2021 Razeware LLC
/// 
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import CoreML

struct UpdatableModel {
  private static var updatedDrawingClassifier: UpdatableDrawingClassifier?
  private static let appDirectory = FileManager.default.urls(
    for: .applicationSupportDirectory,
    in: .userDomainMask).first!
  private static let defaultModelURL =
    UpdatableDrawingClassifier.urlOfModelInThisBundle
  private static var updatedModelURL =
    appDirectory.appendingPathComponent("personalized.mlmodelc")
  private static var tempUpdatedModelURL =
    appDirectory.appendingPathComponent("personalized_tmp.mlmodelc")

  private init() { }

  static var imageConstraint: MLImageConstraint {
    let model = updatedDrawingClassifier ?? UpdatableDrawingClassifier()
    return model.imageConstraint
  }
}

extension UpdatableModel {
  static func predictLabelFor(_ value: MLFeatureValue) -> String? {
    loadModel()
    return updatedDrawingClassifier?.predictLabelFor(value)
  }
  
  static func updateWith(
    trainingData: MLBatchProvider,
    completionHandler: @escaping () -> Void
  ) {
    loadModel()
    UpdatableDrawingClassifier.updateModel(
      at: updatedModelURL,
      with: trainingData) { context in
        saveUpdatedModel(context)
        DispatchQueue.main.async { completionHandler() }
    }
  }
}

private extension UpdatableModel {
  static func saveUpdatedModel(_ updateContext: MLUpdateContext) {
    // 1
    let updatedModel = updateContext.model
    let fileManager = FileManager.default
    do {
      // 2
      try fileManager.createDirectory(
          at: tempUpdatedModelURL,
          withIntermediateDirectories: true,
          attributes: nil)
      // 3
      try updatedModel.write(to: tempUpdatedModelURL)
      // 4
      _ = try fileManager.replaceItemAt(
        updatedModelURL,
        withItemAt: tempUpdatedModelURL)
      print("Updated model saved to:\n\t\(updatedModelURL)")
    } catch let error {
      print("Could not save updated model to the file system: \(error)")
      return
    }
  }
  
  static func loadModel() {
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: updatedModelURL.path) {
      do {
        let updatedModelParentURL =
          updatedModelURL.deletingLastPathComponent()
        try fileManager.createDirectory(
          at: updatedModelParentURL,
          withIntermediateDirectories: true,
          attributes: nil)
        let toTemp = updatedModelParentURL
          .appendingPathComponent(defaultModelURL.lastPathComponent)
        try fileManager.copyItem(
          at: defaultModelURL,
          to: toTemp)
        try fileManager.moveItem(
          at: toTemp,
          to: updatedModelURL)
      } catch {
        print("Error: \(error)")
        return
      }
    }
    guard let model = try? UpdatableDrawingClassifier(
      contentsOf: updatedModelURL) else {
      return
    }
    updatedDrawingClassifier = model
  }
}

extension UpdatableDrawingClassifier {
  
  static func updateModel(
    at url: URL,
    with trainingData: MLBatchProvider,
    completionHandler: @escaping (MLUpdateContext) -> Void
  ) {
    do {
      let updateTask = try MLUpdateTask(
        forModelAt: url,
        trainingData: trainingData,
        configuration: nil,
        completionHandler: completionHandler)
      updateTask.resume()
    } catch {
      print("Couldn't create an MLUpdateTask.")
    }
  }
  
  var imageConstraint: MLImageConstraint {
    return model.modelDescription
      .inputDescriptionsByName["drawing"]!
      .imageConstraint!
  }

  func predictLabelFor(_ value: MLFeatureValue) -> String? {
    guard
      let pixelBuffer = value.imageBufferValue,
      let prediction = try? prediction(drawing: pixelBuffer).label
      else {
         return nil
    }
    if prediction == "unknown" {
      print("No prediction found")
      return nil
    }
    return prediction
  }
}
