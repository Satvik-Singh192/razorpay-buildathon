from backend.db import create_all, get_session
from data.payment_feed_simulator import PaymentFeedSimulator
from data.debtor_reply_simulator import DebtorReplySimulator
from backend.orchestrator import run_full_simulation
from backend.models import Invoice

def main():
    create_all()
    # Assume data is already seeded by python seed.py earlier, but just to be sure we can run it here
    # Actually, the user says they already did prompt 1-7, so the DB should be there, but maybe not seeded
    
    session = get_session()
    
    # Let's see if DB has invoices
    count = session.query(Invoice).count()
    print(f"Invoices in DB: {count}")
    
    if count == 0:
        print("DB is empty. Please run data generator and seeder first.")
        return
        
    payment_feed = PaymentFeedSimulator()
    reply_sim = DebtorReplySimulator()
    
    run_full_simulation(session, payment_feed, reply_sim, num_days=30)
    
if __name__ == "__main__":
    main()
