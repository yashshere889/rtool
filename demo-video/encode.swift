// Turns screencast frames into an H.264 MP4 with AVFoundation (no ffmpeg on this Mac).
//
//   swift encode.swift <frames dir> <out.mp4>
//
// The frames dir holds JPEGs named <ms>.jpg, where <ms> is the frame's time in the finished video, and a file
// END holding the video's length in ms. Each frame stays on screen until the next one's time.
import AVFoundation
import AppKit

let args = CommandLine.arguments
let dir = URL(fileURLWithPath: args[1])
let out = URL(fileURLWithPath: args[2])
let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".jpg") }
let frames = names.compactMap { n -> (Int, URL)? in Int(n.dropLast(4)).map { ($0, dir.appendingPathComponent(n)) } }
  .sorted { $0.0 < $1.0 }
let end = Int(try String(contentsOf: dir.appendingPathComponent("END")).trimmingCharacters(in: .whitespacesAndNewlines))!
guard let first = frames.first, let probe = NSImage(contentsOf: first.1)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
else { fatalError("no frames") }
let width = probe.width & ~1, height = probe.height & ~1  // H.264 wants even sizes

try? FileManager.default.removeItem(at: out)
let writer = try AVAssetWriter(outputURL: out, fileType: .mp4)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
  AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
  AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: Int(ProcessInfo.processInfo.environment["BITRATE"] ?? "") ?? 1_000_000, AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel] as [String: Any],
])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
  kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB, kCVPixelBufferWidthKey as String: width,
  kCVPixelBufferHeightKey as String: height,
])
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

func buffer(_ url: URL) -> CVPixelBuffer {
  let image = NSImage(contentsOf: url)!.cgImage(forProposedRect: nil, context: nil, hints: nil)!
  var pb: CVPixelBuffer?
  CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
  CVPixelBufferLockBaseAddress(pb!, [])
  let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb!), width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: CVPixelBufferGetBytesPerRow(pb!), space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
  ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
  CVPixelBufferUnlockBaseAddress(pb!, [])
  return pb!
}

// A still screen is re-sent every 2 s so players can seek into it.
var times: [(Int, URL)] = []
for (i, f) in frames.enumerated() {
  let next = i + 1 < frames.count ? frames[i + 1].0 : end
  var t = f.0
  while t < next { times.append((t, f.1)); t += 2000 }
}
times.append((end - 40, frames.last!.1))  // players end the video at the last frame, not at endSession
var last: Int = -1
for (t, url) in times where t > last {
  while !input.isReadyForMoreMediaData { usleep(2000) }
  adaptor.append(buffer(url), withPresentationTime: CMTime(value: CMTimeValue(t), timescale: 1000))
  last = t
}
input.markAsFinished()
writer.endSession(atSourceTime: CMTime(value: CMTimeValue(end), timescale: 1000))
let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
if writer.status != .completed { fatalError("write failed: \(String(describing: writer.error))") }
print("wrote \(out.path): \(times.count) frames, \(end / 1000) s, \(width)x\(height)")
