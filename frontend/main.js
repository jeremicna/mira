async function getState() {
	try {
		const res = await fetch("http://localhost:8000/api/state");
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		console.log("State:", data);
		return data;
	} catch (err) {
		console.error("Failed to fetch state:", err);
	}
}

async function main() {
	while (true) {
		await getState();
		
		await new Promise(resolve => setTimeout(resolve, 200));
	}
}

main();
