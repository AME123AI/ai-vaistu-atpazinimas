                "**Veiklioji medžiaga:**",
                ingredient_1 or "—"
            )

            if first_selected:
                st.write("**Stiprumas:**", first_selected.get("strength") or "nenurodyta")
                st.write("**Farmacinė forma:**", first_selected.get("form") or "nenurodyta")
                st.caption("Duomenys iš konkretaus patvirtinto VVKT kandidato.")
            elif first_row is not None:
                st.write("**Stiprumas:**", clean_value(first_row.get("stiprumas")))
                st.write("**Farmacinė forma:**", clean_value(first_row.get("farmacine_forma_lt")))

        with col2:

            st.markdown(
                f"#### {second}"
            )

            st.write(
                "**Veiklioji medžiaga:**",
                ingredient_2 or "—"
            )

            if second_selected:
                st.write("**Stiprumas:**", second_selected.get("strength") or "nenurodyta")
                st.write("**Farmacinė forma:**", second_selected.get("form") or "nenurodyta")
                st.caption("Duomenys iš konkretaus patvirtinto VVKT kandidato.")
            elif second_row is not None:
                st.write("**Stiprumas:**", clean_value(second_row.get("stiprumas")))
                st.write("**Farmacinė forma:**", clean_value(second_row.get("farmacine_forma_lt")))

        # =================================================
        # KAIP VEIKIA KARTU
        # =================================================

        st.markdown(
            "## 🔬 KAIP ŠIE VAISTAI VEIKIA KARTU"
        )

        known_1 = show_ingredient_explanation(
            first,
            ingredient_1
        )

        known_2 = show_ingredient_explanation(
            second,
            ingredient_2
        )

        info_1 = get_ingredient_knowledge(
            ingredient_1
        )

        info_2 = get_ingredient_knowledge(
            ingredient_2
        )

        # Bandomojo CSV tekstai rodomi atskirai nuo porinės išvados.
        with st.expander("📄 VAPRIS 4.5 sąveikų informacija (TESTINĖ)"):
            st.info("Toliau pateikiami atskirų preparatų dokumentų tekstai. "
                    "Tai nėra automatinis šių dviejų vaistų suderinamumo "
                    "patvirtinimas. Prieš klinikinį naudojimą būtina "
                    "patikrinti originalų PCS ir preparato tapatybę.")
            show_test_document_interactions(first)
            st.divider()
            show_test_document_interactions(second)

        # =================================================
        # GRUPIŲ SĄVEIKOS TAISYKLĖ
        # =================================================

        group_rule = find_group_interaction(
            ingredient_1,
            ingredient_2
        )

        if group_rule is not None:

            st.markdown(
                "### ⚠️ Galima farmakologinė sąveika"
            )

            st.warning(
                clean_value(
                    group_rule.get(
                        "interaction"
                    )
                )
            )

            st.markdown(
                "#### Galimas poveikis / rizika"
            )

            st.write(
                clean_value(
                    group_rule.get(
                        "effect"
                    )
                )
            )

            st.markdown(
                "#### 📋 Išvada"
            )

            st.error(
                clean_value(
                    group_rule.get(
                        "conclusion"
                    )
                )
            )

            source = clean_value(
                group_rule.get(
                    "source"
                )
            )

            if source != "—":

                st.markdown(
                    "#### 📚 Sąveikos šaltinis"
                )

                st.write(
                    source
                )

        # =================================================
        # JEI GRUPINĖS TAISYKLĖS NĖRA
        # =================================================

        else:

            st.markdown(
                "### 📋 Išvada"
            )

            if known_1 and known_2:

                st.info(
                    "Abiejų veikliųjų medžiagų "
                    "farmakologiniai profiliai "
                    "prototipo bazėje yra aprašyti, "
                    "tačiau šiai konkrečiai "
                    "farmakologinių grupių porai "
                    "patikrinta sąveikos taisyklė "
                    "dar neįtraukta. "
                    "Tai nėra teiginys, kad "
                    "vaistus saugu vartoti kartu."
                )

            else:

                st.warning(
                    "Bent vienos veikliosios "
                    "medžiagos farmakologinis "
                    "profilis dar neįtrauktas "
