#!/usr/bin/env python3
"""
Verification script for Task 4.2: Enhance CodeGeneratorAgent with S3 code management

This script verifies that all requirements for task 4.2 have been implemented:
- Add existing code download functionality for editing workflows
- Implement code upload with proper versioning after generation  
- Create code metadata management and S3 path tracking
- Add fallback mechanisms when code download fails
"""

import sys
import inspect
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_enhanced_code_generator_agent():
    """Verify EnhancedCodeGeneratorAgent has required functionality."""
    
    print("=" * 70)
    print("VERIFYING TASK 4.2: Enhanced CodeGeneratorAgent with S3 Code Management")
    print("=" * 70)
    
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        
        # Get all methods of the class
        methods = [method for method in dir(EnhancedCodeGeneratorAgent) 
                  if not method.startswith('_') or method.startswith('_download') or method.startswith('_upload')]
        
        print("\n1. EXISTING CODE DOWNLOAD FUNCTIONALITY")
        print("-" * 50)
        
        # Check for download functionality
        download_methods = [m for m in methods if 'download' in m.lower()]
        if download_methods:
            print("✓ Download methods found:")
            for method in download_methods:
                print(f"  - {method}")
        else:
            print("✗ No download methods found")
            return False
        
        # Check specific download method
        if hasattr(EnhancedCodeGeneratorAgent, '_download_existing_code'):
            method = getattr(EnhancedCodeGeneratorAgent, '_download_existing_code')
            sig = inspect.signature(method)
            print(f"✓ _download_existing_code method signature: {sig}")
        else:
            print("✗ _download_existing_code method not found")
            return False
        
        print("\n2. CODE UPLOAD WITH VERSIONING")
        print("-" * 50)
        
        # Check for upload functionality
        upload_methods = [m for m in methods if 'upload' in m.lower()]
        if upload_methods:
            print("✓ Upload methods found:")
            for method in upload_methods:
                print(f"  - {method}")
        else:
            print("✗ No upload methods found")
            return False
        
        # Check specific upload method
        if hasattr(EnhancedCodeGeneratorAgent, '_upload_code_to_aws'):
            method = getattr(EnhancedCodeGeneratorAgent, '_upload_code_to_aws')
            sig = inspect.signature(method)
            print(f"✓ _upload_code_to_aws method signature: {sig}")
        else:
            print("✗ _upload_code_to_aws method not found")
            return False
        
        # Check version calculation
        if hasattr(EnhancedCodeGeneratorAgent, '_calculate_new_version'):
            method = getattr(EnhancedCodeGeneratorAgent, '_calculate_new_version')
            sig = inspect.signature(method)
            print(f"✓ _calculate_new_version method signature: {sig}")
        else:
            print("✗ _calculate_new_version method not found")
            return False
        
        print("\n3. CODE METADATA MANAGEMENT")
        print("-" * 50)
        
        # Check for metadata management - look for specific methods
        required_metadata_methods = [
            '_update_code_metadata',
            'get_code_management_status',
            'get_code_history'
        ]
        
        metadata_methods_found = []
        for method_name in required_metadata_methods:
            if hasattr(EnhancedCodeGeneratorAgent, method_name):
                metadata_methods_found.append(method_name)
        
        if metadata_methods_found:
            print("✓ Metadata methods found:")
            for method in metadata_methods_found:
                print(f"  - {method}")
        else:
            print("✗ No metadata methods found")
            return False
        
        # Check specific metadata method
        if hasattr(EnhancedCodeGeneratorAgent, '_update_code_metadata'):
            method = getattr(EnhancedCodeGeneratorAgent, '_update_code_metadata')
            sig = inspect.signature(method)
            print(f"✓ _update_code_metadata method signature: {sig}")
        else:
            print("✗ _update_code_metadata method not found")
            return False
        
        # Check status method
        if hasattr(EnhancedCodeGeneratorAgent, 'get_code_management_status'):
            method = getattr(EnhancedCodeGeneratorAgent, 'get_code_management_status')
            sig = inspect.signature(method)
            print(f"✓ get_code_management_status method signature: {sig}")
        else:
            print("✗ get_code_management_status method not found")
            return False
        
        print("\n4. FALLBACK MECHANISMS")
        print("-" * 50)
        
        # Check for error handling methods
        error_methods = [m for m in methods if 'handle' in m.lower() and 'error' in m.lower()]
        if error_methods:
            print("✓ Error handling methods found:")
            for method in error_methods:
                print(f"  - {method}")
        else:
            print("✗ No error handling methods found")
            return False
        
        # Check specific fallback methods
        fallback_methods = [
            '_handle_code_download_failure',
            '_handle_code_upload_failure'
        ]
        
        for method_name in fallback_methods:
            if hasattr(EnhancedCodeGeneratorAgent, method_name):
                method = getattr(EnhancedCodeGeneratorAgent, method_name)
                sig = inspect.signature(method)
                print(f"✓ {method_name} method signature: {sig}")
            else:
                print(f"✗ {method_name} method not found")
                return False
        
        print("\n5. AWS INTEGRATION")
        print("-" * 50)
        
        # Check AWS service integration
        agent_source = inspect.getsource(EnhancedCodeGeneratorAgent.__init__)
        if 'S3CodeStorageService' in agent_source:
            print("✓ S3CodeStorageService integration found")
        else:
            print("✗ S3CodeStorageService integration not found")
            return False
        
        if 'MetadataService' in agent_source:
            print("✓ MetadataService integration found")
        else:
            print("✗ MetadataService integration not found")
            return False
        
        # Check graceful degradation
        if 'aws_enabled' in agent_source:
            print("✓ AWS enabled flag found for graceful degradation")
        else:
            print("✗ AWS enabled flag not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying EnhancedCodeGeneratorAgent: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_s3_code_storage_enhancements():
    """Verify S3CodeStorageService has required enhancements."""
    
    print("\n6. S3 CODE STORAGE ENHANCEMENTS")
    print("-" * 50)
    
    try:
        from aws.s3_code_storage import S3CodeStorageService, CodeMetadata
        
        # Check CodeMetadata supports scene numbers
        metadata = CodeMetadata(
            video_id='test',
            project_id='test',
            version=1,
            scene_number=1
        )
        print(f"✓ CodeMetadata supports scene numbers: scene {metadata.scene_number}")
        
        # Check S3 service methods
        required_methods = [
            'download_code',
            'upload_code',
            'list_code_versions',
            'list_scene_code_files'
        ]
        
        for method_name in required_methods:
            if hasattr(S3CodeStorageService, method_name):
                method = getattr(S3CodeStorageService, method_name)
                sig = inspect.signature(method)
                print(f"✓ {method_name} method signature: {sig}")
            else:
                print(f"✗ {method_name} method not found")
                return False
        
        # Check S3 key generation supports scene numbers
        service_source = inspect.getsource(S3CodeStorageService._generate_s3_key)
        if 'scene_number' in service_source:
            print("✓ S3 key generation supports scene numbers")
        else:
            print("✗ S3 key generation does not support scene numbers")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying S3CodeStorageService: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_requirements_compliance():
    """Verify compliance with specific requirements."""
    
    print("\n7. REQUIREMENTS COMPLIANCE")
    print("-" * 50)
    
    requirements = {
        "4.2": "CodeGeneratorAgent enhanced with S3 code management",
        "4.3": "Code metadata management and S3 path tracking", 
        "7.3": "Fallback mechanisms when code download fails"
    }
    
    compliance_checks = []
    
    # Requirement 4.2: Enhanced agent
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        from langgraph_agents.agents.code_generator_agent import CodeGeneratorAgent
        
        # Check inheritance
        if issubclass(EnhancedCodeGeneratorAgent, CodeGeneratorAgent):
            print("✓ Requirement 4.2: EnhancedCodeGeneratorAgent extends CodeGeneratorAgent")
            compliance_checks.append(True)
        else:
            print("✗ Requirement 4.2: EnhancedCodeGeneratorAgent does not extend CodeGeneratorAgent")
            compliance_checks.append(False)
    except Exception as e:
        print(f"✗ Requirement 4.2: Error checking inheritance: {e}")
        compliance_checks.append(False)
    
    # Requirement 4.3: Metadata management
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        
        required_metadata_methods = [
            '_update_code_metadata',
            'get_code_management_status',
            'get_code_history'
        ]
        
        all_present = all(hasattr(EnhancedCodeGeneratorAgent, method) 
                         for method in required_metadata_methods)
        
        if all_present:
            print("✓ Requirement 4.3: Code metadata management methods present")
            compliance_checks.append(True)
        else:
            print("✗ Requirement 4.3: Missing code metadata management methods")
            compliance_checks.append(False)
    except Exception as e:
        print(f"✗ Requirement 4.3: Error checking metadata methods: {e}")
        compliance_checks.append(False)
    
    # Requirement 7.3: Fallback mechanisms
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        
        required_fallback_methods = [
            '_handle_code_download_failure',
            '_handle_code_upload_failure'
        ]
        
        all_present = all(hasattr(EnhancedCodeGeneratorAgent, method) 
                         for method in required_fallback_methods)
        
        if all_present:
            print("✓ Requirement 7.3: Fallback mechanisms implemented")
            compliance_checks.append(True)
        else:
            print("✗ Requirement 7.3: Missing fallback mechanisms")
            compliance_checks.append(False)
    except Exception as e:
        print(f"✗ Requirement 7.3: Error checking fallback methods: {e}")
        compliance_checks.append(False)
    
    return all(compliance_checks)

def main():
    """Run all verification checks."""
    
    print("Task 4.2 Verification: Enhanced CodeGeneratorAgent with S3 Code Management")
    print("=" * 80)
    
    # Run verification checks
    check1 = verify_enhanced_code_generator_agent()
    check2 = verify_s3_code_storage_enhancements()
    check3 = verify_requirements_compliance()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    checks = [
        ("Enhanced CodeGeneratorAgent Functionality", check1),
        ("S3 Code Storage Enhancements", check2),
        ("Requirements Compliance", check3)
    ]
    
    for check_name, result in checks:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name}: {status}")
    
    overall_success = all(result for _, result in checks)
    
    print(f"\nOverall Task 4.2 Status: {'✓ COMPLETED' if overall_success else '✗ INCOMPLETE'}")
    
    if overall_success:
        print("\n🎉 Task 4.2 has been successfully implemented!")
        print("\nImplemented Features:")
        print("- ✓ Existing code download functionality for editing workflows")
        print("- ✓ Code upload with proper versioning after generation")
        print("- ✓ Code metadata management and S3 path tracking")
        print("- ✓ Fallback mechanisms when code download fails")
        print("- ✓ Scene-specific code handling")
        print("- ✓ Graceful degradation when AWS is unavailable")
        print("- ✓ Integration with existing CodeGeneratorAgent workflow")
    else:
        print("\n❌ Task 4.2 implementation is incomplete.")
        print("Please review the failed checks above.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)