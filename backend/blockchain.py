import hashlib
import json
import time
import requests
import os



class Block:
    def __init__(self, index, timestamp, request_id, plastic_type, confidence, image_hash, verification_status, previous_hash):
        
        self.index = index
        self.timestamp = timestamp
        self.request_id = request_id
        self.plastic_type = plastic_type
        self.confidence = confidence
        self.image_hash = image_hash
        self.verification_status = verification_status
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()


    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "plastic_type": self.plastic_type,
            "confidence": self.confidence,
            "image_hash": self.image_hash,
            "verification_status": self.verification_status,
            "previous_hash": self.previous_hash
        }, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "plastic_type": self.plastic_type,
            "confidence": self.confidence,
            "image_hash": self.image_hash,
            "verification_status": self.verification_status,
            "previous_hash": self.previous_hash,
            "hash": self.hash
        }


class Blockchain:
    def __init__(self, blockchain_file="blockchain.json"):
        self.blockchain_file = blockchain_file
        self.nodes_file = blockchain_file.replace(".json", "_nodes.json")
        self.chain = []
        self.nodes = set()
        self.load_chain()
        self.load_nodes()

        if not self.chain:
            self.create_genesis_block()
            self.save_chain()

        self.auto_sync()

    def create_genesis_block(self):
        genesis = Block(index=0, 
                        timestamp=time.time(), 
                        request_id="GENESIS", 
                        plastic_type="N/A", 
                        confidence=0.0, 
                        image_hash="0", 
                        verification_status="N/A", 
                        previous_hash="0")
        self.chain.append(genesis)

    def add_block_from_data(self, record):
        last_block = self.chain[-1]

        new_block = Block(
            index=len(self.chain),
            timestamp=record["timestamp"],
            request_id=record["request_id"],
            plastic_type=record["plastic_type"],
            confidence=record["confidence"],
            image_hash=record["image_hash"],
            verification_status=record["verification_status"],
            previous_hash=last_block.hash
        )

        self.chain.append(new_block)
        self.save_chain()
        self.auto_sync()

    def save_chain(self):
        with open(self.blockchain_file, "w") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=4)

    def load_chain(self):
        if os.path.exists(self.blockchain_file):
            with open(self.blockchain_file, "r") as f:
                data = json.load(f)
                self.chain = []

                for block_data in data:
                    block = Block(
                        block_data["index"],
                        block_data["timestamp"],
                        block_data["request_id"],
                        block_data["plastic_type"],
                        block_data["confidence"],
                        block_data["image_hash"],
                        block_data["verification_status"],
                        block_data["previous_hash"]
                    )
                    block.hash = block_data["hash"]
                    self.chain.append(block)
    
    def register_node(self, address):
        self.nodes.add(address)
        self.save_nodes()
    
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.calculate_hash():
                print(f"Invalid hash at block {i}")
                return False
            if current.previous_hash != previous.hash:
                print(f"Broken chain at block {i}")
                return False

        return True
    
    def resolve_conflicts(self, force=False):
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)

        for node in neighbours:
            try:
                response = requests.get(f"{node}/chain")

                if response.status_code == 200:
                    chain_data = response.json()
                    length = len(chain_data)

                    if self.is_external_chain_valid(chain_data):
                        if force:
                            new_chain = chain_data
                            break
                        if length > max_length:
                            max_length = length
                            new_chain = chain_data
            except:
                continue

        if new_chain:
            self.replace_chain(new_chain)
            return True
        return False
    
    def is_external_chain_valid(self, chain_data):
        for i in range(1, len(chain_data)):
            current = chain_data[i]
            previous = chain_data[i - 1]

            block_string = json.dumps({
                "index": current["index"],
                "timestamp": current["timestamp"],
                "request_id": current["request_id"],
                "plastic_type": current["plastic_type"],
                "confidence": current["confidence"],
                "image_hash": current["image_hash"],
                "verification_status": current["verification_status"],
                "previous_hash": current["previous_hash"]
            }, sort_keys=True)

            recalculated_hash = hashlib.sha256(block_string.encode()).hexdigest()

            if current["hash"] != recalculated_hash:
                return False

            if current["previous_hash"] != previous["hash"]:
                return False

        return True
    
    def replace_chain(self, chain_data):
        self.chain = []
        for block_data in chain_data:
            block = Block(
                block_data["index"],
                block_data["timestamp"],
                block_data["request_id"],
                block_data["plastic_type"],
                block_data["confidence"],
                block_data["image_hash"],
                block_data["verification_status"],
                block_data["previous_hash"]
            )
            block.hash = block_data["hash"]
            self.chain.append(block)

        self.save_chain()
    
    def auto_sync(self):
        if not self.is_chain_valid():
            print("Local chain invalid. Syncing from peers...")
            replaced = self.resolve_conflicts(force=True)
            if replaced:
                print("Chain replaced successfully.")
            else:
                print("No valid peer chain found.")
        else:
            self.resolve_conflicts()
    
    def save_nodes(self):
        with open(self.nodes_file, "w") as f:
            json.dump(list(self.nodes), f)

    def load_nodes(self):
        if os.path.exists(self.nodes_file):
            with open(self.nodes_file, "r") as f:
                self.nodes = set(json.load(f))
    
    


