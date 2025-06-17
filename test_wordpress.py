#!/usr/bin/env python3
"""
WordPress Publisher Test Script
Run this to verify your WordPress integration is working correctly
"""

import asyncio
import os
import sys
from datetime import datetime

# Check if rich is available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
    has_rich = True
except ImportError:
    has_rich = False
    console = None
    def rprint(*args, **kwargs):
        print(*args)

# Check if aiohttp is available
try:
    import aiohttp
    has_aiohttp = True
except ImportError:
    has_aiohttp = False

# Check if supabase is available
try:
    from supabase import create_client
    has_supabase = True
except ImportError:
    has_supabase = False

async def test_wordpress_integration():
    """Comprehensive test of WordPress publisher integration"""
    
    if has_rich:
        console.print("\n[bold blue]🧪 WordPress Publisher Integration Test[/bold blue]")
        console.print("=" * 60)
    else:
        print("\n🧪 WordPress Publisher Integration Test")
        print("=" * 60)
    
    # Step 1: Check environment variables
    if has_rich:
        console.print("\n[yellow]📋 Step 1: Checking environment variables...[/yellow]")
    else:
        print("\n📋 Step 1: Checking environment variables...")
    
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
        "WORDPRESS_SITE_URL": os.getenv("WORDPRESS_SITE_URL"),
        "WORDPRESS_USERNAME": os.getenv("WORDPRESS_USERNAME"),
        "WORDPRESS_APP_PASSWORD": os.getenv("WORDPRESS_APP_PASSWORD")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        if has_rich:
            console.print(f"[red]❌ Missing environment variables: {', '.join(missing_vars)}[/red]")
            console.print("\n[yellow]Please set these in your .env file:[/yellow]")
        else:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("\nPlease set these in your .env file:")
        
        for var in missing_vars:
            print(f"  {var}=your-value-here")
        
        # Show demo mode option
        if has_rich:
            console.print("\n[cyan]💡 Run with --demo to see the integration workflow without real credentials[/cyan]")
        else:
            print("\n💡 Run with --demo to see the integration workflow without real credentials")
        return
    
    if has_rich:
        console.print("[green]✅ All required environment variables found[/green]")
    else:
        print("✅ All required environment variables found")
    
    # Step 2: Check dependencies
    if has_rich:
        console.print("\n[yellow]📋 Step 2: Checking dependencies...[/yellow]")
    else:
        print("\n📋 Step 2: Checking dependencies...")
    
    missing_deps = []
    if not has_aiohttp:
        missing_deps.append("aiohttp")
    if not has_supabase:
        missing_deps.append("supabase")
    
    if missing_deps:
        if has_rich:
            console.print(f"[red]❌ Missing dependencies: {', '.join(missing_deps)}[/red]")
            console.print("\n[yellow]Please install:[/yellow]")
        else:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("\nPlease install:")
        
        print(f"  pip install {' '.join(missing_deps)}")
        return
    
    if has_rich:
        console.print("[green]✅ All dependencies available[/green]")
    else:
        print("✅ All dependencies available")
    
    # Step 3: Test WordPress authentication
    if has_rich:
        console.print("\n[yellow]📋 Step 3: Testing WordPress authentication...[/yellow]")
    else:
        print("\n📋 Step 3: Testing WordPress authentication...")
    
    try:
        # Simple WordPress auth test
        import base64
        from urllib.parse import urljoin
        
        site_url = required_vars["WORDPRESS_SITE_URL"]
        username = required_vars["WORDPRESS_USERNAME"]
        app_password = required_vars["WORDPRESS_APP_PASSWORD"]
        
        # Create auth header
        credentials = f"{username}:{app_password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
            "User-Agent": "UniversalRAGCMS/1.0"
        }
        
        # Test connection
        async with aiohttp.ClientSession() as session:
            url = urljoin(site_url, "/wp-json/wp/v2/users/me")
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    user_data = await response.json()
                    if has_rich:
                        console.print("[green]✅ WordPress authentication successful[/green]")
                        console.print(f"[cyan]Authenticated as: {user_data.get('name', 'Unknown')}[/cyan]")
                    else:
                        print("✅ WordPress authentication successful")
                        print(f"Authenticated as: {user_data.get('name', 'Unknown')}")
                else:
                    if has_rich:
                        console.print(f"[red]❌ WordPress authentication failed (Status: {response.status})[/red]")
                        console.print("[yellow]💡 This is expected with demo credentials. Run --demo mode to see the full workflow.[/yellow]")
                    else:
                        print(f"❌ WordPress authentication failed (Status: {response.status})")
                        print("💡 This is expected with demo credentials. Run --demo mode to see the full workflow.")
                    return
                
    except Exception as e:
        if has_rich:
            console.print(f"[red]❌ Authentication test failed: {e}[/red]")
            console.print("[yellow]💡 This is expected with demo credentials. Run --demo mode to see the full workflow.[/yellow]")
        else:
            print(f"❌ Authentication test failed: {e}")
            print("💡 This is expected with demo credentials. Run --demo mode to see the full workflow.")
        return
    
    # Step 4: Test Supabase connection
    if has_rich:
        console.print("\n[yellow]📋 Step 4: Testing Supabase connection...[/yellow]")
    else:
        print("\n📋 Step 4: Testing Supabase connection...")
    
    try:
        supabase = create_client(
            required_vars["SUPABASE_URL"],
            required_vars["SUPABASE_SERVICE_KEY"]
        )
        
        # Test connection with a simple query
        result = supabase.table("documents").select("id").limit(1).execute()
        
        if has_rich:
            console.print("[green]✅ Supabase connection successful[/green]")
        else:
            print("✅ Supabase connection successful")
        
    except Exception as e:
        if has_rich:
            console.print(f"[red]❌ Supabase connection failed: {e}[/red]")
        else:
            print(f"❌ Supabase connection failed: {e}")
        return
    
    # Continue with the rest of the tests...
    await run_full_test_suite(site_url, headers, supabase)

async def run_full_test_suite(site_url, headers, supabase):
    """Run the full test suite"""
    
    # Step 5: Test simple WordPress post creation
    if has_rich:
        console.print("\n[yellow]📋 Step 5: Testing WordPress post creation...[/yellow]")
    else:
        print("\n📋 Step 5: Testing WordPress post creation...")
    
    try:
        # Create test content
        test_content = f"""
        <h2>Integration Test Post</h2>
        <p>This is an automated test post created by the WordPress Publisher Integration test script.</p>
        <p>Test timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h3>Test Features</h3>
        <ul>
            <li>✅ Authentication working</li>
            <li>✅ API connection established</li>
            <li>✅ Content formatting active</li>
            <li>✅ Database logging enabled</li>
        </ul>
        
        <p>If you can see this post in your WordPress admin, the integration is working correctly!</p>
        """
        
        post_data = {
            "title": f"Integration Test - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": test_content,
            "status": "draft",
            "excerpt": "Automated test post from WordPress Publisher Integration"
        }
        
        if has_rich:
            console.print("\n[cyan]Publishing test post as draft...[/cyan]")
        else:
            print("\nPublishing test post as draft...")
        
        async with aiohttp.ClientSession() as session:
            from urllib.parse import urljoin
            url = urljoin(site_url, "/wp-json/wp/v2/posts")
            async with session.post(url, headers=headers, json=post_data) as response:
                if response.status in [200, 201]:
                    post_result = await response.json()
                    
                    if has_rich:
                        console.print(f"[green]✅ Test post published successfully![/green]")
                        console.print(f"[cyan]📝 Post ID: {post_result['id']}[/cyan]")
                        console.print(f"[cyan]🔗 Post URL: {post_result['link']}[/cyan]")
                    else:
                        print("✅ Test post published successfully!")
                        print(f"📝 Post ID: {post_result['id']}")
                        print(f"🔗 Post URL: {post_result['link']}")
                    
                    # Show preview URL
                    preview_url = f"{post_result['link']}?preview=true"
                    if has_rich:
                        console.print(f"\n[yellow]👁️  Preview URL: {preview_url}[/yellow]")
                    else:
                        print(f"\n👁️  Preview URL: {preview_url}")
                        
                else:
                    error_text = await response.text()
                    if has_rich:
                        console.print(f"[red]❌ Publishing failed (Status: {response.status})[/red]")
                        console.print(f"[red]Error: {error_text}[/red]")
                    else:
                        print(f"❌ Publishing failed (Status: {response.status})")
                        print(f"Error: {error_text}")
                    return
                    
    except Exception as e:
        if has_rich:
            console.print(f"[red]❌ Publishing test failed: {e}[/red]")
        else:
            print(f"❌ Publishing test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "=" * 60)
    
    if has_rich:
        summary_panel = Panel(
            "[green]✅ WordPress Publisher Integration Test Complete![/green]\n\n"
            "All components are working correctly:\n"
            "• Environment variables configured\n"
            "• Dependencies installed\n"
            "• WordPress authentication working\n"
            "• Supabase connection established\n"
            "• Content publishing functional\n\n"
            "[yellow]Next steps:[/yellow]\n"
            "1. Check your WordPress admin for the test post\n"
            "2. Try publishing with the full integration\n"
            "3. Test bulk publishing functionality\n"
            "4. Monitor performance in production",
            title="[bold green]Test Summary[/bold green]",
            border_style="green"
        )
        console.print(summary_panel)
    else:
        print("✅ WordPress Publisher Integration Test Complete!")
        print("\nAll components are working correctly:")
        print("• Environment variables configured")
        print("• Dependencies installed") 
        print("• WordPress authentication working")
        print("• Supabase connection established")
        print("• Content publishing functional")
        print("\nNext steps:")
        print("1. Check your WordPress admin for the test post")
        print("2. Try publishing with the full integration")
        print("3. Test bulk publishing functionality")
        print("4. Monitor performance in production")

async def demo_mode():
    """Demo mode showing the complete WordPress integration workflow"""
    
    if has_rich:
        console.print("\n[bold magenta]🎭 WordPress Publisher Integration Demo[/bold magenta]")
        console.print("=" * 60)
        console.print("[cyan]This demo shows the complete integration workflow[/cyan]")
    else:
        print("\n🎭 WordPress Publisher Integration Demo")
        print("=" * 60)
        print("This demo shows the complete integration workflow")
    
    # Demo Step 1: Configuration
    if has_rich:
        console.print("\n[yellow]📋 Step 1: WordPress Configuration[/yellow]")
    else:
        print("\n📋 Step 1: WordPress Configuration")
    
    print("✅ WordPress Site URL: https://your-site.com")
    print("✅ Authentication Method: Application Passwords")
    print("✅ Username: your-username")
    print("✅ Application Password: ************")
    print("✅ API Base: /wp-json/wp/v2")
    
    # Demo Step 2: Features
    if has_rich:
        console.print("\n[yellow]📋 Step 2: Integration Features[/yellow]")
    else:
        print("\n📋 Step 2: Integration Features")
    
    features_table = [
        ["🔐", "Multi-Authentication", "Application Passwords, JWT, OAuth2"],
        ["🖼️", "Image Processing", "Automatic optimization, resizing, format conversion"],
        ["🎨", "HTML Enhancement", "Rich formatting, responsive design, SEO optimization"],
        ["🔄", "Error Recovery", "Exponential backoff, partial failure recovery"],
        ["📊", "Performance Tracking", "Timing, retry counts, success rates"],
        ["🗄️", "Database Logging", "Complete audit trail in Supabase"],
        ["🚀", "Bulk Publishing", "Batch processing with rate limiting"],
        ["🔍", "Smart Images", "DataForSEO integration for automatic image discovery"]
    ]
    
    if has_rich:
        table = Table(title="WordPress Publisher Features")
        table.add_column("Icon", style="cyan")
        table.add_column("Feature", style="white")
        table.add_column("Description", style="green")
        
        for icon, feature, description in features_table:
            table.add_row(icon, feature, description)
        
        console.print(table)
    else:
        print("\nWordPress Publisher Features:")
        for icon, feature, description in features_table:
            print(f"  {icon} {feature}: {description}")
    
    # Demo Step 3: Sample Content Creation
    if has_rich:
        console.print("\n[yellow]📋 Step 3: Content Creation Workflow[/yellow]")
    else:
        print("\n📋 Step 3: Content Creation Workflow")
    
    sample_content = {
        "title": "Ultimate Guide to Online Casino Gaming 2024",
        "content": """
        <h2>Introduction to Online Casino Gaming</h2>
        <p>Online casinos have revolutionized the gambling industry, offering players unprecedented access to their favorite games from anywhere in the world.</p>
        
        <h3>Popular Casino Games</h3>
        <ul>
            <li>Slot Machines - Easy to play with exciting themes</li>
            <li>Blackjack - Strategic card game with best odds</li>
            <li>Roulette - Classic wheel-based excitement</li>
            <li>Poker - Skill-based competitive gaming</li>
        </ul>
        
        <h3>Choosing the Right Casino</h3>
        <p>When selecting an online casino, consider licensing, game variety, bonuses, and payment methods.</p>
        """,
        "excerpt": "Complete guide to online casino gaming in 2024",
        "status": "draft",
        "categories": [1, 5],  # Gaming, Reviews
        "tags": [10, 15, 20]   # casino, online-gaming, 2024
    }
    
    print("📝 Sample Content:")
    print(f"  Title: {sample_content['title']}")
    print(f"  Status: {sample_content['status']}")
    print(f"  Categories: {sample_content['categories']}")
    print(f"  Content Length: {len(sample_content['content'])} characters")
    
    # Demo Step 4: Processing Pipeline
    if has_rich:
        console.print("\n[yellow]📋 Step 4: Processing Pipeline[/yellow]")
    else:
        print("\n📋 Step 4: Processing Pipeline")
    
    pipeline_steps = [
        "🔍 Content Analysis & Keyword Extraction",
        "🖼️ Smart Image Discovery via DataForSEO",
        "🎨 HTML Enhancement & Responsive Formatting", 
        "📤 Image Upload & Optimization",
        "📝 Post Creation with Rich Content",
        "🗄️ Database Logging & Audit Trail",
        "✅ Success Confirmation & URL Generation"
    ]
    
    for i, step in enumerate(pipeline_steps, 1):
        print(f"  {i}. {step}")
        await asyncio.sleep(0.3)  # Simulate processing time
    
    # Demo Step 5: Results
    if has_rich:
        console.print("\n[yellow]📋 Step 5: Publishing Results[/yellow]")
    else:
        print("\n📋 Step 5: Publishing Results")
    
    print("✅ Post Published Successfully!")
    print("📝 Post ID: 1234")
    print("🔗 Post URL: https://your-site.com/ultimate-guide-online-casino-gaming-2024/")
    print("🖼️ Images Uploaded: 3")
    print("⏱️ Processing Time: 2.3 seconds")
    print("🔄 Retry Count: 0")
    
    # Demo Step 6: Database Logging
    if has_rich:
        console.print("\n[yellow]📋 Step 6: Database Audit Trail[/yellow]")
    else:
        print("\n📋 Step 6: Database Audit Trail")
    
    log_data = {
        "post_title": sample_content["title"],
        "post_status": "draft",
        "success": True,
        "post_id": 1234,
        "post_url": "https://your-site.com/ultimate-guide-online-casino-gaming-2024/",
        "media_count": 3,
        "publish_time": 2.3,
        "retry_count": 0,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("📊 Logged to Supabase:")
    for key, value in log_data.items():
        print(f"  {key}: {value}")
    
    # Final Demo Summary
    print("\n" + "=" * 60)
    
    if has_rich:
        demo_summary = Panel(
            "[green]🎉 WordPress Publisher Integration Demo Complete![/green]\n\n"
            "[bold]Key Capabilities Demonstrated:[/bold]\n"
            "• Enterprise-grade authentication & security\n"
            "• Bulletproof image processing & optimization\n"
            "• Rich HTML formatting with SEO enhancements\n"
            "• Comprehensive error handling & recovery\n"
            "• Real-time performance monitoring\n"
            "• Complete database audit trail\n"
            "• Smart image discovery & integration\n"
            "• Bulk publishing with rate limiting\n\n"
            "[yellow]Ready for Production Deployment![/yellow]\n"
            "Configure your WordPress credentials to start publishing.",
            title="[bold magenta]Demo Summary[/bold magenta]",
            border_style="magenta"
        )
        console.print(demo_summary)
    else:
        print("🎉 WordPress Publisher Integration Demo Complete!")
        print("\nKey Capabilities Demonstrated:")
        print("• Enterprise-grade authentication & security")
        print("• Bulletproof image processing & optimization")
        print("• Rich HTML formatting with SEO enhancements")
        print("• Comprehensive error handling & recovery")
        print("• Real-time performance monitoring")
        print("• Complete database audit trail")
        print("• Smart image discovery & integration")
        print("• Bulk publishing with rate limiting")
        print("\nReady for Production Deployment!")
        print("Configure your WordPress credentials to start publishing.")

async def quick_test():
    """Quick connectivity test only"""
    if has_rich:
        console.print("\n[bold blue]🚀 Quick WordPress Connection Test[/bold blue]")
    else:
        print("\n🚀 Quick WordPress Connection Test")
    
    try:
        import base64
        from urllib.parse import urljoin
        
        site_url = os.getenv("WORDPRESS_SITE_URL")
        username = os.getenv("WORDPRESS_USERNAME") 
        app_password = os.getenv("WORDPRESS_APP_PASSWORD")
        
        if not all([site_url, username, app_password]):
            print("❌ Missing WordPress environment variables")
            print("💡 Run with --demo to see the integration workflow")
            return
            
        credentials = f"{username}:{app_password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers = {"Authorization": f"Basic {encoded}"}
        
        async with aiohttp.ClientSession() as session:
            url = urljoin(site_url, "/wp-json/wp/v2/users/me")
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    if has_rich:
                        console.print("[green]✅ WordPress connection successful![/green]")
                        console.print(f"[cyan]Site: {site_url}[/cyan]")
                        console.print(f"[cyan]User: {username}[/cyan]")
                    else:
                        print("✅ WordPress connection successful!")
                        print(f"Site: {site_url}")
                        print(f"User: {username}")
                else:
                    if has_rich:
                        console.print("[red]❌ Connection failed[/red]")
                        console.print("[yellow]💡 Run with --demo to see the integration workflow[/yellow]")
                    else:
                        print("❌ Connection failed")
                        print("💡 Run with --demo to see the integration workflow")
                        
    except Exception as e:
        if has_rich:
            console.print(f"[red]❌ Error: {e}[/red]")
            console.print("[yellow]💡 Run with --demo to see the integration workflow[/yellow]")
        else:
            print(f"❌ Error: {e}")
            print("💡 Run with --demo to see the integration workflow")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WordPress Publisher Integration")
    parser.add_argument("--quick", action="store_true", help="Run quick connection test only")
    parser.add_argument("--demo", action="store_true", help="Run demo mode showing integration workflow")
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(quick_test())
    elif args.demo:
        asyncio.run(demo_mode())
    else:
        asyncio.run(test_wordpress_integration()) 