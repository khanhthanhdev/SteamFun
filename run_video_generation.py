#!/usr/bin/env python3
"""
Video Generation Runner
Practical script to run video generation using the working legacy system.
This demonstrates the complete user input to video output workflow.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def generate_video_with_legacy_system(topic: str, description: str, only_plan: bool = False):
    """Generate video using the legacy system that we know works."""
    
    print("🎬 Starting Video Generation with Legacy System")
    print("=" * 70)
    
    try:
        from generate_video import VideoGenerationConfig, EnhancedVideoGenerator
        
        # Create configuration
        config = VideoGenerationConfig(
            planner_model="openai/gpt-4o-mini",
            scene_model="openai/gpt-4o-mini",
            helper_model="openai/gpt-4o-mini",
            output_dir="generated_videos",
            verbose=True,
            use_rag=False,  # Keep simple for demo
            use_langfuse=False,  # Keep simple for demo
            use_visual_fix_code=False,  # Keep simple for demo
            max_scene_concurrency=1,  # Process scenes sequentially
            max_retries=3,
            enable_caching=True,
            preview_mode=True  # Faster rendering for demo
        )
        
        print("✅ Configuration created")
        print(f"   Model: {config.planner_model}")
        print(f"   Output directory: {config.output_dir}")
        print(f"   Preview mode: {config.preview_mode}")
        
        # Create generator
        print("\n🏗️ Initializing video generator...")
        generator = EnhancedVideoGenerator(config)
        print("✅ Generator initialized successfully")
        
        # Show input parameters
        print(f"\n📝 Input Parameters:")
        print(f"   Topic: {topic}")
        print(f"   Description: {description}")
        print(f"   Planning only: {only_plan}")
        
        # Generate video
        print(f"\n🚀 Starting video generation pipeline...")
        
        if only_plan:
            # Just generate the plan
            print("📋 Generating scene outline...")
            scene_outline = await generator.generate_scene_outline(topic, description)
            
            if scene_outline:
                print("✅ Scene outline generated successfully!")
                print(f"   Length: {len(scene_outline)} characters")
                
                # Show preview
                lines = scene_outline.split('\n')[:15]  # First 15 lines
                print("\n📄 Scene Outline Preview:")
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                
                if len(scene_outline.split('\n')) > 15:
                    print("   ... (truncated)")
                
                print(f"\n✅ Planning completed! Check output directory for full outline.")
                return True
            else:
                print("❌ Failed to generate scene outline")
                return False
        else:
            # Full video generation
            print("🎬 Running complete video generation pipeline...")
            await generator.generate_video_pipeline(
                topic=topic,
                description=description,
                only_plan=False
            )
            
            print("✅ Video generation completed!")
            print(f"   Check the '{config.output_dir}' directory for your video")
            return True
            
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_argument_parser():
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Generate educational videos using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a complete video
  python run_video_generation.py --topic "Python Variables" --description "Tutorial on string, int, and boolean variables"
  
  # Generate planning only (faster)
  python run_video_generation.py --topic "Math Basics" --description "Addition and subtraction" --plan-only
  
  # Interactive mode
  python run_video_generation.py --interactive
        """
    )
    
    parser.add_argument("--topic", help="Video topic")
    parser.add_argument("--description", help="Video description")
    parser.add_argument("--plan-only", action="store_true", help="Generate planning only (no video rendering)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode - prompts for input")
    
    return parser


def get_interactive_input():
    """Get input from user interactively."""
    
    print("🎯 Interactive Video Generation")
    print("=" * 50)
    
    print("Please provide the following information:")
    
    topic = input("\n📝 Video Topic: ").strip()
    if not topic:
        print("❌ Topic cannot be empty")
        return None, None, False
    
    description = input("📄 Video Description: ").strip()
    if not description:
        print("❌ Description cannot be empty")
        return None, None, False
    
    plan_only_input = input("📋 Generate planning only? (y/N): ").strip().lower()
    plan_only = plan_only_input in ['y', 'yes']
    
    print(f"\n✅ Input received:")
    print(f"   Topic: {topic}")
    print(f"   Description: {description}")
    print(f"   Plan only: {plan_only}")
    
    confirm = input("\n🚀 Proceed with video generation? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Generation cancelled by user")
        return None, None, False
    
    return topic, description, plan_only


async def main():
    """Main function."""
    
    print("🎬 AI Video Generation System")
    print("Transforms your ideas into educational videos")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        print("Please ensure your .env file contains:")
        print("OPENAI_API_KEY=your-key-here")
        return 1
    
    print("✅ Environment check passed")
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Get input parameters
    if args.interactive:
        topic, description, plan_only = get_interactive_input()
        if not topic:
            return 1
    elif args.topic and args.description:
        topic = args.topic
        description = args.description
        plan_only = args.plan_only
    else:
        print("❌ Please provide --topic and --description, or use --interactive mode")
        parser.print_help()
        return 1
    
    # Generate video
    try:
        success = await generate_video_with_legacy_system(topic, description, plan_only)
        
        if success:
            print("\n🎉 SUCCESS! Video generation completed!")
            
            if plan_only:
                print("\n💡 Next steps:")
                print("   1. Review the generated scene outline")
                print("   2. Run again without --plan-only to generate the full video")
                print("   3. Check the output directory for all files")
            else:
                print("\n💡 Your video is ready!")
                print("   1. Check the 'generated_videos' directory")
                print("   2. Look for the combined MP4 file")
                print("   3. Individual scene files are also available")
            
            return 0
        else:
            print("\n❌ Video generation failed")
            print("Check the error messages above for details")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)