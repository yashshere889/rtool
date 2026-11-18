// Pulls stills out of a finished MP4, to check what viewers will see:  swift stills.swift <in.mp4> <out dir> <s> [<s> ...]
import AVFoundation
import AppKit

let a = CommandLine.arguments
let asset = AVURLAsset(url: URL(fileURLWithPath: a[1]))
let gen = AVAssetImageGenerator(asset: asset)
gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 5)
gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 5)
print("duration", CMTimeGetSeconds(asset.duration))
for s in a.dropFirst(3) {
  let image = try gen.copyCGImage(at: CMTime(seconds: Double(s)!, preferredTimescale: 600), actualTime: nil)
  let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])!
  try png.write(to: URL(fileURLWithPath: "\(a[2])/still-\(s).png"))
}
