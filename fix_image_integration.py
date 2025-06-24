#!/usr/bin/env python3
"""
Quick fix for BulletproofImageIntegrator None comparison issue
"""

import re

def fix_bulletproof_image_integrator():
    """Fix the None comparison issue in BulletproofImageIntegrator"""
    
    file_path = "src/integrations/bulletproof_image_integrator.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the width/height None comparison issue
        old_pattern = r'width = img\.get\(\'width\', 0\)\s*\n\s*height = img\.get\(\'height\', 0\)'
        new_pattern = 'width = img.get(\'width\', 0) or 0\n            height = img.get(\'height\', 0) or 0'
        
        content = re.sub(old_pattern, new_pattern, content)
        
        # Also fix any other potential None comparisons
        content = content.replace(
            "img.get('quality_score', 0) > 0.7",
            "(img.get('quality_score', 0) or 0) > 0.7"
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed BulletproofImageIntegrator None comparison issues")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing BulletproofImageIntegrator: {e}")
        return False

if __name__ == "__main__":
    fix_bulletproof_image_integrator() 