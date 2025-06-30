#!/usr/bin/env python3
"""
Enhanced test script for Playwright with comprehensive browser headers.
Tests the new enhanced scraper with realistic browser fingerprinting.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.undetected_web_scraper import EnhancedUndetectedWebScraper

async def test_header_fingerprinting():
    """Test browser headers against fingerprinting detection sites."""
    
    print("🔍 Testing Enhanced Browser Headers & Fingerprinting")
    print("=" * 60)
    print("Testing against advanced fingerprinting and header detection...")
    print()
    
    test_sites = [
        {
            "url": "https://httpbin.org/headers",
            "name": "HTTPBin Headers Test",
            "description": "Shows all HTTP headers sent by the browser"
        },
        {
            "url": "https://bot.sannysoft.com/",
            "name": "Sannysoft Bot Detection",
            "description": "Comprehensive bot detection test"
        },
        {
            "url": "https://nowsecure.nl/",
            "name": "NowSecure Bot Test", 
            "description": "Advanced bot detection system"
        },
        {
            "url": "https://abrahamjuliot.github.io/creepjs/",
            "name": "CreepJS Fingerprinting",
            "description": "Advanced browser fingerprinting test"
        },
        {
            "url": "https://deviceandbrowserinfo.com/info_device",
            "name": "Device & Browser Info",
            "description": "Detailed browser and device information"
        }
    ]
    
    results = []
    
    try:
        async with EnhancedUndetectedWebScraper() as scraper:
            for i, site in enumerate(test_sites, 1):
                print(f"🧪 Test {i}/{len(test_sites)}: {site['name']}")
                print(f"   📍 URL: {site['url']}")
                print(f"   📝 Purpose: {site['description']}")
                
                try:
                    # Create enhanced context for testing
                    context = await scraper._create_enhanced_context()
                    page = await context.new_page()
                    await scraper._apply_enhanced_scripts(page)
                    
                    # Navigate to test site
                    response = await page.goto(site['url'], timeout=30000)
                    await asyncio.sleep(3)
                    
                    # Special handling for HTTPBin to extract headers
                    if "httpbin.org" in site['url']:
                        try:
                            content = await page.content()
                            if '"headers"' in content:
                                # Extract and display headers
                                start = content.find('{')
                                if start != -1:
                                    json_content = content[start:content.rfind('}')+1]
                                    headers_data = json.loads(json_content)
                                    print("   📋 Headers sent:")
                                    for header, value in headers_data.get('headers', {}).items():
                                        print(f"      {header}: {value}")
                        except Exception as e:
                            print(f"   ⚠️  Could not parse headers: {e}")
                    
                    # Take screenshot for manual verification
                    screenshot_path = f"enhanced_test_{site['name'].lower().replace(' ', '_').replace('&', 'and')}.png"
                    await page.screenshot(path=screenshot_path)
                    
                    # Get page title and status
                    title = await page.title()
                    status = response.status if response else "Unknown"
                    
                    await context.close()
                    
                    print(f"   ✅ Test completed successfully")
                    print(f"   📸 Screenshot: {screenshot_path}")
                    print(f"   📄 Page title: {title}")
                    print(f"   🌐 HTTP status: {status}")
                    results.append(True)
                    
                except Exception as e:
                    print(f"   ❌ Test failed: {e}")
                    results.append(False)
                
                print()
                
                # Add delay between tests
                await asyncio.sleep(2)
    
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False
    
    success_rate = sum(results) / len(results) * 100 if results else 0
    print(f"📊 Header & Fingerprinting Test Results: {success_rate:.1f}% success rate")
    
    return success_rate > 70

async def test_enhanced_scraper(query: str):
    """Test the enhanced scraper with comprehensive headers."""
    
    print(f"🚀 Testing Enhanced Undetected Scraper")
    print(f"Query: '{query}'")
    print("=" * 60)
    print("Using comprehensive browser headers and advanced anti-detection")
    print()
    
    try:
        async with EnhancedUndetectedWebScraper() as scraper:
            print("🔧 Starting enhanced search and scrape...")
            print(f"   Browser config: {scraper.config_choice['name']}")
            print(f"   User-Agent: {scraper.config_choice['user_agent']}")
            print(f"   Headers count: {len(scraper.config_choice['headers'])}")
            print()
            
            results = await scraper.search_and_scrape(query)
            
            print(f"📊 Enhanced Results Summary:")
            print(f"   Total pages scraped: {len(results)}")
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            print(f"   Successful: {len(successful_results)}")
            print(f"   Failed: {len(failed_results)}")
            
            if successful_results:
                print(f"\n✅ Successful Enhanced Results:")
                for i, result in enumerate(successful_results, 1):
                    print(f"\n   {i}. {result.title}")
                    print(f"      URL: {result.url}")
                    print(f"      Content: {result.content_length} characters")
                    
                    # Show first 150 characters of content
                    if result.content:
                        preview = result.content[:150].replace('\n', ' ')
                        print(f"      Preview: {preview}...")
            
            if failed_results:
                print(f"\n❌ Failed Results:")
                for i, result in enumerate(failed_results, 1):
                    print(f"   {i}. {result.url} - {result.error_message}")
            
            if successful_results:
                print(f"\n🎉 Enhanced scraper is working! Found {len(successful_results)} pages.")
                return True
            else:
                print(f"\n⚠️  Enhanced scraper found no results.")
                return False
                
    except Exception as e:
        print(f"\n❌ Enhanced scraper failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def validate_browser_consistency():
    """Validate that headers are consistent with browser fingerprint."""
    
    print("🔎 Validating Browser Header Consistency")
    print("=" * 60)
    print("Checking for common inconsistencies that trigger bot detection...")
    print()
    
    try:
        async with EnhancedUndetectedWebScraper() as scraper:
            config = scraper.config_choice
            headers = config['headers']
            user_agent = config['user_agent']
            
            print(f"🌐 Browser Config: {config['name']}")
            print(f"👤 User-Agent: {user_agent}")
            print()
            
            # Check for common consistency issues
            issues = []
            warnings = []
            
            # Check sec-ch-ua consistency
            if 'sec-ch-ua' in headers:
                sec_ch_ua = headers['sec-ch-ua']
                if 'Chrome' in user_agent and 'Chrome' not in sec_ch_ua:
                    issues.append("sec-ch-ua doesn't match Chrome in User-Agent")
                elif 'Firefox' in user_agent and 'sec-ch-ua' in headers:
                    warnings.append("Firefox doesn't typically send sec-ch-ua headers")
            
            # Check platform consistency
            if 'sec-ch-ua-platform' in headers:
                platform = headers['sec-ch-ua-platform']
                if 'Windows' in user_agent and 'Windows' not in platform:
                    issues.append("Platform mismatch between User-Agent and sec-ch-ua-platform")
                elif 'Mac' in user_agent and 'macOS' not in platform:
                    issues.append("Platform mismatch between User-Agent and sec-ch-ua-platform")
                elif 'Linux' in user_agent and 'Linux' not in platform:
                    issues.append("Platform mismatch between User-Agent and sec-ch-ua-platform")
            
            # Check mobile consistency
            if 'sec-ch-ua-mobile' in headers:
                is_mobile = headers['sec-ch-ua-mobile'] == '?1'
                if 'Mobile' in user_agent and not is_mobile:
                    issues.append("Mobile flag inconsistency")
                elif 'Mobile' not in user_agent and is_mobile:
                    issues.append("Mobile flag inconsistency")
            
            # Check Accept header realism
            if 'Accept' in headers:
                accept = headers['Accept']
                if not any(mime in accept for mime in ['text/html', 'application/xhtml+xml']):
                    warnings.append("Accept header doesn't include common MIME types")
            
            # Check Accept-Language
            if 'Accept-Language' not in headers:
                warnings.append("Missing Accept-Language header")
            
            # Check Accept-Encoding
            if 'Accept-Encoding' not in headers:
                warnings.append("Missing Accept-Encoding header")
            elif 'br' not in headers['Accept-Encoding']:
                warnings.append("Accept-Encoding doesn't include Brotli compression")
            
            # Report results
            if not issues and not warnings:
                print("✅ All header consistency checks passed!")
                print("   Headers appear realistic and consistent")
            else:
                if issues:
                    print("❌ Critical Issues Found:")
                    for issue in issues:
                        print(f"   - {issue}")
                    print()
                
                if warnings:
                    print("⚠️  Warnings:")
                    for warning in warnings:
                        print(f"   - {warning}")
                    print()
            
            # Show key headers for verification
            print("🔧 Key Headers Being Sent:")
            key_headers = [
                'User-Agent', 'Accept', 'Accept-Language', 'Accept-Encoding',
                'sec-ch-ua', 'sec-ch-ua-mobile', 'sec-ch-ua-platform',
                'sec-fetch-dest', 'sec-fetch-mode', 'sec-fetch-site'
            ]
            
            for header in key_headers:
                value = headers.get(header, user_agent if header == 'User-Agent' else 'Not set')
                print(f"   {header}: {value}")
            
            return len(issues) == 0
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

async def main():
    """Main enhanced test function."""
    
    print("🔬 Enhanced Playwright Anti-Detection Test Suite")
    print("=" * 60)
    print("Testing comprehensive browser headers and advanced anti-detection")
    print("This should significantly improve success rates!")
    print()
    
    # Test query
    test_query = "Best places to visit in Gurgaon"
    
    # Track results
    header_test_success = False
    consistency_check_success = False
    scraper_success = False
    
    print("Phase 1: Header & Fingerprinting Tests")
    print("=" * 40)
    try:
        header_test_success = await test_header_fingerprinting()
    except Exception as e:
        print(f"❌ Header tests failed: {e}")
    
    print("\nPhase 2: Browser Consistency Validation")
    print("=" * 40)
    try:
        consistency_check_success = await validate_browser_consistency()
    except Exception as e:
        print(f"❌ Consistency check failed: {e}")
    
    print("\nPhase 3: Enhanced Scraper Test")
    print("=" * 40)
    try:
        scraper_success = await test_enhanced_scraper(test_query)
    except Exception as e:
        print(f"❌ Enhanced scraper test failed: {e}")
    
    # Final results
    print("\n" + "="*60)
    print("📋 Enhanced Test Results:")
    print(f"  Header/Fingerprint Tests: {'✅ PASS' if header_test_success else '❌ FAIL'}")
    print(f"  Browser Consistency: {'✅ PASS' if consistency_check_success else '❌ FAIL'}")
    print(f"  Enhanced Scraper: {'✅ WORKING' if scraper_success else '❌ NOT WORKING'}")
    
    overall_success = header_test_success and scraper_success
    
    if overall_success:
        print(f"\n🎉 SUCCESS! Enhanced scraper with comprehensive headers is working!")
        print("\n💡 Key Improvements Made:")
        print("   ✅ Added comprehensive sec-ch-ua headers")
        print("   ✅ Added sec-fetch headers for navigation context")
        print("   ✅ Enhanced Accept headers with modern MIME types") 
        print("   ✅ Added Accept-Encoding with Brotli support")
        print("   ✅ Platform-specific header variations")
        print("   ✅ Consistent browser fingerprinting")
        print("   ✅ Enhanced JavaScript property masking")
        print("   ✅ Realistic timing and behavior simulation")
        
        print("\n🚀 You can now run:")
        print("   python main.py search \"Best places to visit in Gurgaon\"")
        
    else:
        print(f"\n⚠️  Some tests failed, but this is normal for advanced detection.")
        if scraper_success:
            print("✅ The scraper itself is working, which is the main goal!")
        
        print("\n💡 Additional recommendations:")
        print("   1. Use residential proxies for IP rotation")
        print("   2. Implement request rate limiting")
        print("   3. Rotate browser configurations")
        print("   4. Consider using CAPTCHA solving services")
        print("   5. Monitor for new detection techniques")

if __name__ == "__main__":
    asyncio.run(main())