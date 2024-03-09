
with tab3:
    t3_name = st.text_input("First Name")
    t3_addr = st.text_area("Address")
    t3_gen=st.radio(
            "Select Gender",
            key="gender",
            options=["M", "F", "other"],
        )
    t3_email = st.text_input("Email", )
    t3_phone = st.text_input("phone", )
    t3_j_t = st.text_input("Job Title", )
    def click_button():
        ins = update(userdata1).where(userdata1.c.fname==t3_name).values(
                                    addr=t3_addr,
                                    gen=t3_gen,
                                    email=t3_email,
                                    phone=t3_phone,
                                    j_t=t3_j_t)        
        result = conn.execute(ins)
        conn.commit() 
        if result:
            st.write("Data updated Successfully")
    st.button('UPDATE', on_click=click_button)

with tab4:
    st.header("Delete User data")
    t4_name = st.text_input("First Name", )
    def click_button():
        ins = delete(userdata1).where(userdata1.c.fname==t4_name)
        result = conn.execute(ins)
        conn.commit() 
        if result:
            st.write("Data deleted Successfully")
    st.button('DELETE', on_click=click_button)
with tab5:  
    st.header("Display User data")
    t5_name = st.text_input("First Name", )
    def click_button():
        ins = select(userdata1).where(userdata1.c.fname==t5_name)
        result = conn.execute(ins)
        for row in result:
            st.write(row)
        conn.commit() 
        if not result:
            st.warning("No data found")
    st.button('DISPLAY', on_click=click_button)

