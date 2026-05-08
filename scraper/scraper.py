import httpx
from bs4 import BeautifulSoup
import json
import asyncio
import os
import re

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/?type=2"

async def get_page_content(url: str, client: httpx.AsyncClient):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = await client.get(url, headers=headers, follow_redirects=True)
    return response.text

def parse_catalog_page(html: str):
    soup = BeautifulSoup(html, "lxml")
    products = []
    
    # SHL catalog items are typically in anchor tags under certain classes, but we can look for the links
    links = soup.find_all("a", href=re.compile(r"/products/product-catalog/view/[^/]+/$"))
    for link in links:
        # Avoid things that might not be product links
        name = link.get_text(separator=' ', strip=True)
        if not name or len(name) < 3:
            continue
            
        url = link["href"]
        if not url.startswith("http"):
            url = BASE_URL + url
        products.append({"name": name, "url": url})
        
    # Deduplicate by url
    unique_products = {p["url"]: p for p in products}.values()
    
    # Find next page link
    next_url = None
    next_link = soup.find("a", string=re.compile(r"Next", re.I))
    if not next_link:
        # try checking aria-label or classes if text match fails
        next_link = soup.find("a", attrs={"aria-label": re.compile(r"Next", re.I)})
        
    if next_link and next_link.get("href"):
        href = next_link["href"]
        if not href.startswith("http"):
            href = BASE_URL + href
        next_url = href
        
    return list(unique_products), next_url

async def parse_product_detail(url: str, client: httpx.AsyncClient):
    html = await get_page_content(url, client)
    soup = BeautifulSoup(html, "lxml")
    
    data = {
        "url": url,
        "description": "",
        "test_type": "",
        "job_levels": [],
        "languages": 0,
        "remote_testing": False,
        "adaptive_irt": False,
        "duration_minutes": None
    }
    
    # Try to find Description
    desc_headers = soup.find_all(re.compile(r"^h[1-6]$"), string=re.compile(r"Description", re.I))
    for header in desc_headers:
        p = header.find_next_sibling("p")
        if p:
            data["description"] = p.get_text(strip=True)
            break
            
    # Job levels
    job_headers = soup.find_all(re.compile(r"^h[1-6]$"), string=re.compile(r"Job levels", re.I))
    for header in job_headers:
        content = header.find_next_sibling()
        if content:
            levels = [l.strip() for l in content.get_text().split(",") if l.strip()]
            data["job_levels"] = levels
            break
            
    # Languages
    lang_headers = soup.find_all(re.compile(r"^h[1-6]$"), string=re.compile(r"Languages", re.I))
    for header in lang_headers:
        content = header.find_next_sibling()
        if content:
            langs = [l.strip() for l in content.get_text().split(",") if l.strip()]
            data["languages"] = len(langs)
            break
            
    # Assessment length
    len_headers = soup.find_all(re.compile(r"^h[1-6]$"), string=re.compile(r"Assessment length", re.I))
    for header in len_headers:
        content = header.find_next_sibling()
        if content:
            text = content.get_text(separator=" ")
            
            match = re.search(r"minutes\s*=\s*(\d+)", text, re.I)
            if match:
                data["duration_minutes"] = int(match.group(1))
                
            type_match = re.search(r"Test Type:\s*([A-Z\s,]+)", text, re.I)
            if type_match:
                # keep only uppercase letters (excluding spaces and commas)
                data["test_type"] = "".join(re.findall(r"[A-Z]", type_match.group(1)))
                
            if "Remote Testing:" in text:
                data["remote_testing"] = True # Based on common pattern or presence
            break

    # If test_type is missing, maybe it's listed differently
    if not data["test_type"]:
        type_span = soup.find(string=re.compile(r"Test Type:", re.I))
        if type_span and type_span.parent:
            text = type_span.parent.get_text()
            match = re.search(r"Test Type:\s*([A-Z\s,]+)", text, re.I)
            if match:
                data["test_type"] = "".join(re.findall(r"[A-Z]", match.group(1)))

    full_text = soup.get_text(separator=" ").lower()
    if "adaptive" in full_text or "irt" in full_text:
        data["adaptive_irt"] = True
        
    return data

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        all_products = []
        current_url = CATALOG_URL
        pages_scraped = 0
        
        print(f"Starting scrape from {current_url}...")
        while current_url:
            print(f"Scraping catalog page: {current_url}")
            html = await get_page_content(current_url, client)
            products, next_url = parse_catalog_page(html)
            
            print(f"Found {len(products)} products on page {pages_scraped + 1}")
            
            for i, p in enumerate(products):
                # Avoid duplicates
                if any(existing["url"] == p["url"] for existing in all_products):
                    continue
                
                print(f"Scraping product {i+1}/{len(products)}: {p['name']}")
                try:
                    details = await parse_product_detail(p["url"], client)
                    # merge data
                    for k, v in details.items():
                        if k not in p or not p[k]:
                            p[k] = v
                    all_products.append(p)
                except Exception as e:
                    print(f"Error scraping {p['url']}: {e}")
                
                await asyncio.sleep(0.5) # rate limiting
                
            current_url = next_url
            pages_scraped += 1
            
        os.makedirs("data", exist_ok=True)
        with open("data/catalog.json", "w", encoding="utf-8") as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        print(f"Done! Scraped {len(all_products)} assessments.")

if __name__ == "__main__":
    asyncio.run(main())
